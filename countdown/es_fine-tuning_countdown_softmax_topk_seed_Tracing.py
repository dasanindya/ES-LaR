import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from softmax_topk_new import SoftmaxTopKNew
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import wandb

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="ES Countdown with SoftmaxTopKNew (Seed Tracing)")
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU')
parser.add_argument(
    '--verbose',
    action='store_true',
    help='Verbose logs: ES threads, and SoftmaxTopKNew generate path + per-thought detail (first seq in batch only)',
)
parser.add_argument('--data_sample', type=int, default=1000, help='Number of data samples for training')
parser.add_argument('--output_dir', type=str, default='.', help='Base directory to save the final model and checkpoints')
parser.add_argument('--wandb_project', type=str, default='es-countdown-softmax-topk-new', help='Weights & Biases project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name (optional)')
parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (optional)')
parser.add_argument('--eval_dataset_size', type=int, default=None, help='Number of samples for final evaluation')
parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
parser.add_argument('--num_latent_thoughts', type=int, default=4, help='Latent steps before answer (0 = plain batched HF generate)')
parser.add_argument('--blend_top_k', type=int, default=10, help='Top-k vocab entries for latent embedding blend')
parser.add_argument('--initial_seed', type=int, default=33, help='Random seed for ES (default: 33)')
parser.add_argument('--num_iterations', type=int, default=10, help='Number of ES outer iterations')
parser.add_argument('--population_size', type=int, default=10, help='Population size (seeds per iteration)')
parser.add_argument('--sigma', type=float, default=0.001, help='ES perturbation noise scale')
parser.add_argument('--alpha', type=float, default=0.0005, help='ES learning rate')
args = parser.parse_args()


NUM_ITERATIONS = args.num_iterations
POPULATION_SIZE = args.population_size
SIGMA = float(args.sigma)
ALPHA = float(args.alpha)
max_new_tokens = 128
do_sample = False
initial_seed = args.initial_seed

from countdown_task import reward_function
print("Using countdown reward function with SoftmaxTopKNew (Qwen/Llama causal LM) + seed tracing")


def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, args, dataset_size):
    save_dir = os.path.join(
        args.output_dir,
        f"{model_name.replace('/', '_')}_softmaxtopknew_es_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_n{dataset_size}_checkpoint",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.base_causallm.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Checkpoint saved successfully.")


def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """Generate with SoftmaxTopKNew: batched `generate` for both num_latent_thoughts==0 (HF) and >0 (latent+decode)."""
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    use_fast = isinstance(model, SoftmaxTopKNew) and model._can_fast_generate()

    generated_texts = []
    with torch.inference_mode():
        if use_fast:
            if verbose:
                print(
                    f"[evaluate_model] proc={accelerator.process_index} thread={thread_id} seed={seed_idx} → "
                    "SoftmaxTopKNew fast path: batched base_causallm.generate (num_latent_thoughts=0)"
                )
            tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
            input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
            attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                verbose_latent=verbose,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
            for i in range(len(input_texts)):
                try:
                    gen_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                except TypeError:
                    tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
                    filtered = [t for t in tokens if t is not None]
                    gen_text = tokenizer.convert_tokens_to_string(filtered)
                generated_texts.append(gen_text)
            del input_ids, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if verbose:
                print(
                    f"[evaluate_model] proc={accelerator.process_index} thread={thread_id} seed={seed_idx} → "
                    f"SoftmaxTopKNew.generate() batched latent path "
                    f"(num_latent_thoughts={model.num_latent_thoughts}, blend_top_k={model.blend_top_k}, "
                    f"n={len(input_texts)} seqs); verbose_latent={'batch[0] only' if verbose else 'off'}"
                )
            tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
            input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
            attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                verbose_latent=verbose,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(accelerator.device)
            for i in range(len(input_texts)):
                try:
                    gen_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                except TypeError:
                    tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
                    filtered = [t for t in tokens if t is not None]
                    gen_text = tokenizer.convert_tokens_to_string(filtered)
                generated_texts.append(gen_text)
            del input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rewards = []
    for gen_text, tgt_text, inp_text in zip(generated_texts, target_texts, input_texts):
        numbers = None
        target = None
        if "[" in inp_text and "]" in inp_text:
            start_idx = inp_text.find("[")
            end_idx = inp_text.find("]")
            if start_idx != -1 and end_idx != -1:
                numbers_str = inp_text[start_idx + 1 : end_idx]
                numbers = [int(n) for n in numbers_str.split() if n.isdigit()]
        if tgt_text.isdigit():
            target = int(tgt_text)

        model_response = gen_text
        if "assistant:" in gen_text:
            model_response = gen_text.split("assistant:")[-1].strip()

        reward_result = reward_function(model_response, numbers, target)
        rewards.append(reward_result["reward"])

    if return_text:
        return rewards, generated_texts
    return rewards


def process_seed(seed_args):
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, dataset = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Apply weight perturbation deterministically from `seed`
    for _, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    input_texts = [x for x, _ in dataset]
    target_texts = [y for _, y in dataset]
    rewards = evaluate_model(
        model, tokenizer, input_texts, target_texts, accelerator,
        seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False,
    )
    total_reward = sum(rewards)

    # Restore original weights
    for _, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed))
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {total_reward / len(dataset):.4f}")

    return seed_idx, total_reward / len(dataset)


def main():
    accelerator = Accelerator()

    data_path = os.path.join(os.path.dirname(__file__), 'data/countdown.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with open(data_path, 'r') as f:
        data_json = json.load(f)

    dataset = [(item['context'], item['target']) for item in data_json]
    dataset = dataset[: args.data_sample]

    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} countdown samples from {data_path}")
        print(
            f"Using SoftmaxTopKNew + seed tracing: latent_thoughts={args.num_latent_thoughts}, blend_top_k={args.blend_top_k}"
        )

    # Initialize wandb on main process
    if accelerator.is_main_process and not args.disable_wandb:
        try:
            wandb_kwargs = {"project": args.wandb_project}
            wandb_kwargs["name"] = args.wandb_run_name or f"{args.model_name.replace('/', '_')}_softmaxtopknew_seed{initial_seed}_pop{POPULATION_SIZE}"
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            wandb.init(**wandb_kwargs)
            wandb.config.update({
                "sigma": SIGMA,
                "alpha": ALPHA,
                "population_size": POPULATION_SIZE,
                "num_iterations": NUM_ITERATIONS,
                "precision": args.precision,
                "gpu_threads": args.gpu_threads,
                "model": "SoftmaxTopKNew",
                "initial_seed": initial_seed,
                "num_latent_thoughts": args.num_latent_thoughts,
                "blend_top_k": args.blend_top_k,
                "seed_tracing": True,
            })
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")

    if accelerator.is_main_process:
        print(f"Processes: {accelerator.num_processes}, GPU threads: {args.gpu_threads}")
        print(f"Population: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}, Sigma: {SIGMA}, Alpha: {ALPHA}")

    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name} with SoftmaxTopKNew wrapper...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    eos_token_id = tokenizer.eos_token_id

    model_list = []
    for _ in range(args.gpu_threads):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        )
        model_list.append(
            SoftmaxTopKNew(
                base_model,
                eos_token_id,
                num_latent_thoughts=args.num_latent_thoughts,
                blend_top_k=args.blend_top_k,
                verbose=args.verbose,
            )
        )

    if accelerator.is_main_process:
        print("Model loaded successfully (SoftmaxTopKNew wrapper)")
        if args.num_latent_thoughts <= 0:
            gen_note = "HF batched generate (no latent blend steps)"
        else:
            gen_note = (
                f"latent path (top-{args.blend_top_k} softmax blend per thought), then greedy decode"
            )
        print(
            "[setup] Generation: wrapped model is always SoftmaxTopKNew; "
            f"num_latent_thoughts={args.num_latent_thoughts} → {gen_note}"
        )

    for model in model_list:
        model.eval()

    force_memory_cleanup()
    training_start_time = time.time()
    np.random.seed(initial_seed)

    # Seed tracing / resume reward caching
    seed_trace_dir = os.path.join(args.output_dir, "seed_tracing")
    # Trace file selection logic:
    # If multiple traces exist for the same hyperparameter combo but different
    # iteration counts, we want to pick the trace with the *most* iterations
    # so requesting fewer iterations can still reuse cached rewards.
    trace_prefix_start = (
        f"seed_trace"
        f"_{args.model_name.replace('/', '_')}"
        f"_pop{POPULATION_SIZE}"
        f"_sigma{SIGMA}"
        f"_alpha{ALPHA}"
        f"_seed{initial_seed}"
        f"_iter"
    )
    trace_suffix_end = (
        f"_nt{args.num_latent_thoughts}"
        f"_k{args.blend_top_k}"
        f"_ds{args.data_sample}"
        f".json"
    )

    def _trace_filename_for_iter(iter_count: int) -> str:
        return f"{trace_prefix_start}{iter_count}{trace_suffix_end}"

    requested_trace_filename = _trace_filename_for_iter(NUM_ITERATIONS)
    requested_seed_trace_path = os.path.join(seed_trace_dir, requested_trace_filename)

    best_iter = -1
    best_seed_trace_path = None
    if os.path.exists(seed_trace_dir):
        for fname in os.listdir(seed_trace_dir):
            if not fname.startswith(trace_prefix_start):
                continue
            if not fname.endswith(trace_suffix_end):
                continue
            # fname = prefix_start + <iter_count> + suffix_end
            mid = fname[len(trace_prefix_start) : -len(trace_suffix_end)]
            try:
                iter_count = int(mid)
            except ValueError:
                continue
            if iter_count > best_iter:
                best_iter = iter_count
                best_seed_trace_path = os.path.join(seed_trace_dir, fname)

    # Load the best-available trace (max iteration). If none exist, start fresh.
    seed_trace_path = best_seed_trace_path if best_seed_trace_path is not None else requested_seed_trace_path

    def _init_seed_trace():
        return {
            "header": {
                "model_name": args.model_name,
                "hyperparameters": {
                    "population_size": POPULATION_SIZE,
                    "sigma": SIGMA,
                    "alpha": ALPHA,
                    "initial_seed": initial_seed,
                    "max_new_tokens": max_new_tokens,
                    "num_iterations": NUM_ITERATIONS,
                    "num_latent_thoughts": args.num_latent_thoughts,
                    "blend_top_k": args.blend_top_k,
                    "data_sample": args.data_sample,
                },
                "execution_history": [],
                "metadata": {
                    "description": "Seed trace for ES fine-tuning (SoftmaxTopKNew)",
                    "data_structure": "seed_value -> iteration_number -> {seed_idx, reward, normalized_reward, positive_contribution, timestamp, job_id}",
                },
            },
            "data": {},
        }

    # Load trace on all ranks so caching works correctly
    seed_trace = None
    try:
        if seed_trace_path is not None and os.path.exists(seed_trace_path):
            with open(seed_trace_path, "r") as f:
                seed_trace = json.load(f)
    except Exception:
        seed_trace = None

    if seed_trace is None:
        seed_trace = _init_seed_trace()

    # If we loaded a trace with fewer iterations than requested, we will write
    # updates into the requested trace file so future runs can reuse it.
    if best_iter >= 0 and best_iter < NUM_ITERATIONS:
        seed_trace_path = requested_seed_trace_path

    # Ensure header hyperparameters reflect *this* run's requested iterations.
    # This doesn't affect reward caching, but makes the trace file self-consistent.
    try:
        seed_trace["header"]["hyperparameters"]["population_size"] = POPULATION_SIZE
        seed_trace["header"]["hyperparameters"]["sigma"] = SIGMA
        seed_trace["header"]["hyperparameters"]["alpha"] = ALPHA
        seed_trace["header"]["hyperparameters"]["initial_seed"] = initial_seed
        seed_trace["header"]["hyperparameters"]["num_iterations"] = NUM_ITERATIONS
        seed_trace["header"]["hyperparameters"]["num_latent_thoughts"] = args.num_latent_thoughts
        seed_trace["header"]["hyperparameters"]["blend_top_k"] = args.blend_top_k
        seed_trace["header"]["hyperparameters"]["data_sample"] = args.data_sample
        seed_trace["header"]["hyperparameters"]["max_new_tokens"] = max_new_tokens
    except Exception:
        pass

    if accelerator.is_main_process:
        os.makedirs(seed_trace_dir, exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        seed_trace["header"]["execution_history"].append({
            "job_id": job_id,
            "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "iterations_completed": 0,
        })
        # Write immediately so the run is visible in the trace file.
        try:
            with open(seed_trace_path, "w") as f:
                json.dump(seed_trace, f, indent=2)
        except Exception:
            print("Warning: failed to write initial seed trace file.")

    # ES loop
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        local_seeds = [(seed_idx, seed) for seed_idx, seed in enumerate(seeds)
                       if seed_idx % accelerator.num_processes == accelerator.process_index]

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds")

        local_rewards = []
        local_uncached_seeds = []
        iter_key = str(iteration + 1)

        # Check cache for seeds that already exist in the trace
        for seed_idx, seed in local_seeds:
            seed_val = str(seed)
            cached = (
                seed_val in seed_trace.get("data", {}) and
                iter_key in seed_trace["data"].get(seed_val, {}) and
                "reward" in seed_trace["data"][seed_val][iter_key]
            )
            if cached:
                local_rewards.append((seed_idx, float(seed_trace["data"][seed_val][iter_key]["reward"])))
            else:
                local_uncached_seeds.append((seed_idx, seed))

        # Evaluate uncached seeds
        if len(local_uncached_seeds) > 0:
            batch_size = max(1, min(args.gpu_threads, len(local_uncached_seeds)))
            for batch_start in range(0, len(local_uncached_seeds), batch_size):
                batch_seeds = local_uncached_seeds[batch_start : batch_start + batch_size]
                with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                    thread_args = [
                        (seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose, dataset)
                        for thread_id, (seed_idx, seed) in enumerate(batch_seeds)
                    ]
                    results = list(executor.map(process_seed, thread_args))
                    local_rewards.extend(results)
                force_memory_cleanup()

        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")

        original_model = model_list[0]
        for _, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                gen.manual_seed(int(seeds[seed_idx]))
                noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                noise.mul_(float(rewards_normalized[seed_idx]))
                update.add_(noise)
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            for name, param in model_list[model_idx].named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        # Update seed trace file and cache results (main process only)
        if accelerator.is_main_process:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            job_id = os.environ.get("SLURM_JOB_ID", "unknown")

            seed_trace.setdefault("data", {})
            for seed_idx in range(POPULATION_SIZE):
                seed_val = str(seeds[seed_idx])
                seed_entry = seed_trace["data"].setdefault(seed_val, {})
                seed_entry[iter_key] = {
                    "seed_idx": seed_idx,
                    "reward": float(rewards_tensor[seed_idx]),
                    "normalized_reward": round(float(rewards_normalized[seed_idx]), 6),
                    "positive_contribution": bool(rewards_normalized[seed_idx] > 0),
                    "timestamp": timestamp,
                    "job_id": job_id,
                }

            # Mark execution history progress
            if seed_trace.get("header", {}).get("execution_history"):
                seed_trace["header"]["execution_history"][-1]["iterations_completed"] = iteration + 1

            try:
                with open(seed_trace_path, "w") as f:
                    json.dump(seed_trace, f, indent=2)
            except Exception:
                print("Warning: failed to write seed trace file.")

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            if not args.disable_wandb:
                try:
                    wandb.log({
                        "iteration": iteration + 1,
                        "mean_reward": mean_reward,
                        "min_reward": min_reward,
                        "max_reward": max_reward,
                        "iter_time": iter_time,
                    })
                except Exception:
                    pass

            if (iteration + 1) % 100 == 0:
                save_model_checkpoint(original_model, tokenizer, iteration + 1, model_name, initial_seed, args, len(dataset))

    total_time = time.time() - training_start_time

    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        save_dir = os.path.join(
            args.output_dir,
            f"{args.model_name.replace('/', '_')}_softmaxtopknew_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_n{len(dataset)}_final",
        )
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving final model to {save_dir}...")
        original_model.base_causallm.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Final model saved successfully.")


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()

