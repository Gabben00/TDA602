import torch
import random
import json
import os
import gc
import argparse
import subprocess
import sys

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
#from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import destroy_model_parallel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run refusal direction pipeline (debug-friendly).")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')

    # --- Dataset sizes (keep tiny for debugging) ---
    parser.add_argument('--n_train', type=int, default=8,
                        help='Training samples per class (default: 8 for debug)')
    parser.add_argument('--n_val',   type=int, default=4,
                        help='Validation samples per class (default: 4 for debug)')
    parser.add_argument('--n_test',  type=int, default=4,
                        help='Test samples (default: 4 for debug)')

    # --- Skip flags: run only the stages you want to test ---
    parser.add_argument('--skip_filter',      action='store_true',
                        help='Skip refusal-score filtering of train/val sets')
    parser.add_argument('--skip_completions', action='store_true',
                        help='Skip generation of completions (stages 3a, 4a)')
    parser.add_argument('--skip_eval',        action='store_true',
                        help='Skip jailbreak/refusal evaluation (stages 3b, 4b)')
    parser.add_argument('--skip_loss',        action='store_true',
                        help='Skip cross-entropy loss evaluation (stage 5)')

    # --- Generation / loss knobs ---
    parser.add_argument('--max_new_tokens',     type=int, default=20,
                        help='Max tokens to generate per completion (default: 20 for debug)')
    parser.add_argument('--ce_loss_batch_size', type=int, default=1,
                        help='Batch size for CE loss eval (default: 1)')
    parser.add_argument('--ce_loss_n_batches',  type=int, default=2,
                        help='Number of batches for CE loss eval (default: 2)')

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    random.seed(42)
    harmful_train  = random.sample(load_dataset_split(harmtype='harmful',  split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val    = random.sample(load_dataset_split(harmtype='harmful',  split='val',   instructions_only=True), cfg.n_val)
    harmless_val   = random.sample(load_dataset_split(harmtype='harmless', split='val',   instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores  = get_refusal_scores(model_base.model, harmful_train,  model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train  = filter_examples(harmful_train,  harmful_train_scores,  0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores  = get_refusal_scores(model_base.model, harmful_val,  model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val  = filter_examples(harmful_val,  harmful_val_scores,  0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val



def _eval_worker(cfg_artifact_path, dataset_name, label, eval_methodologies):
    import json
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
    path = f'{cfg_artifact_path}/completions/{dataset_name}_{label}_completions.json'
    with open(path) as f:
        completions = json.load(f)
    evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=f'{cfg_artifact_path}/completions/{dataset_name}_{label}_evaluations.json'
    )

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'), exist_ok=True)
    mean_diffs = generate_directions(
        model_base, harmful_train, harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))
    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))
    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'), exist_ok=True)
    pos, layer, direction = select_direction(
        model_base, harmful_val, harmless_val, candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"))
    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)
    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')
    return pos, layer, direction


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks,
                                               intervention_label, dataset_name, dataset=None):
    os.makedirs(os.path.join(cfg.artifact_path(), 'completions'), exist_ok=True)
    if dataset is None:
        dataset = load_dataset(dataset_name)
    completions = model_base.generate_completions(
        dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens)
    out = f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json'
    with open(out, "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    p = mp.Process(
        target=_eval_worker,
        args=(cfg.artifact_path(), dataset_name, intervention_label, eval_methodologies)
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Eval worker failed with exit code {p.exitcode}")


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'), exist_ok=True)
    on_dist = os.path.join(cfg.artifact_path(), 'completions/harmless_baseline_completions.json')
    loss_evals = evaluate_loss(
        model_base, fwd_pre_hooks, fwd_hooks,
        batch_size=cfg.ce_loss_batch_size,
        n_batches=cfg.ce_loss_n_batches,
        completions_file_path=on_dist)
    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)



def run_pipeline(args):
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    # Override Config with CLI values so downstream calls use debug sizes
    cfg.n_train             = args.n_train
    cfg.n_val               = args.n_val
    cfg.n_test              = args.n_test
    cfg.max_new_tokens      = args.max_new_tokens
    cfg.ce_loss_batch_size  = args.ce_loss_batch_size
    cfg.ce_loss_n_batches   = args.ce_loss_n_batches

    # Disable filtering when --skip_filter is passed
    if args.skip_filter:
        cfg.filter_train = False
        cfg.filter_val   = False

    print(f"[debug] n_train={cfg.n_train}  n_val={cfg.n_val}  n_test={cfg.n_test}")
    print(f"[debug] max_new_tokens={cfg.max_new_tokens}  filter_train={cfg.filter_train}")

    model_base = construct_model_base(cfg.model_path)

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # Stage 1 & 2: always run (cheap with tiny data)
    print(f"[debug] GPU memory before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"[debug] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    print("[stage 1/5] Generating candidate directions...")
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    print(f"[debug] GPU memory before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"[debug] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    print("[stage 2/5] Selecting direction...")
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)
    print(f"[debug] Selected pos={pos}  layer={layer}")

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks,   actadd_fwd_hooks   = [
        (model_base.model_block_modules[layer],
         get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # Stage 3a & 4a: generation
    print(f"[debug] GPU memory before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"[debug] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    if not args.skip_completions:
        print("[stage 3a/5] Generating completions on harmful eval datasets...")
        for dataset_name in cfg.evaluation_datasets:
            for label, pre, hooks in [
                ('baseline', baseline_fwd_pre_hooks, baseline_fwd_hooks),
                ('ablation', ablation_fwd_pre_hooks, ablation_fwd_hooks),
                ('actadd',   actadd_fwd_pre_hooks,   actadd_fwd_hooks),
            ]:
                generate_and_save_completions_for_dataset(cfg, model_base, pre, hooks, label, dataset_name)

        print("[stage 4a/5] Generating completions on harmless test set...")
        harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks,
                                                  'baseline', 'harmless', dataset=harmless_test)
        actadd_refusal_pre_hooks = [
            (model_base.model_block_modules[layer],
             get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))]
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, [],
                                                  'actadd', 'harmless', dataset=harmless_test)
    else:
        print("[stage 3a/4a] Skipped (--skip_completions).")

    print(f"[debug] GPU memory before cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"[debug] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

    # Stage 5: loss
    #print(f"[debug] direction device={direction.device}, has_nan={direction.isnan().any()}")
    if not args.skip_loss:
        print("[stage 5????/5] Evaluating loss...")
        model_base = construct_model_base(cfg.model_path)
        for label, pre, hooks in [
            ('baseline', baseline_fwd_pre_hooks, baseline_fwd_hooks),
            ('ablation', ablation_fwd_pre_hooks, ablation_fwd_hooks),
            ('actadd',   actadd_fwd_pre_hooks,   actadd_fwd_hooks),
        ]:
            evaluate_loss_for_datasets(cfg, model_base, pre, hooks, label)
    else:
        print("[stage 5] Skipped (--skip_loss).")

    print("[done]")
    #model_base.model.cpu()
    del model_base.model
    del model_base
    gc.collect()
    destroy_model_parallel()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    import ctypes
    libcudart = ctypes.CDLL('libcudart.so')
    libcudart.cudaDeviceReset()
    print(f"[debug] GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    print(f"[debug] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    # Stage 3b & 4b: evaluation
    if not args.skip_eval:
        print("[stage 3b/5] Evaluating completions on harmful eval datasets...")
        for dataset_name in cfg.evaluation_datasets:
            for label in ['baseline', 'ablation', 'actadd']:
                evaluate_completions_and_save_results_for_dataset(
                    cfg, label, dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        print("[stage 4b/5] Evaluating completions on harmless test set...")
        for label in ['baseline', 'actadd']:
            evaluate_completions_and_save_results_for_dataset(
                cfg, label, 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    else:
        print("[stage 3b/4b] Skipped (--skip_eval).")

if __name__ == "__main__":
    mp.set_start_method('spawn')   # ← 'spawn' gives a clean CUDA context, 'fork' does not
    args = parse_arguments()
    run_pipeline(args)
"""if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)"""
