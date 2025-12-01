"""
Custom model wrapper for R-Sparse evaluation with lighteval 0.12.0
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List, Union

import torch
from transformers import AutoTokenizer, AutoConfig

from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.utils.cache_management import SampleCache

from utils.setup import setup_model, setup_config


class RSparseModel(LightevalModel):
    """Custom lighteval model wrapper for R-Sparse models."""

    def __init__(self, config, env_config=None):
        self._model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B")
        self._method = config.get("method", "full")
        self._config_file = config.get("config_file", None)
        self._sparse_config_file = config.get("sparse_config_file", None)
        self._cache_dir = config.get("cache_dir", None)
        self._device = config.get("device", "cuda:0")
        self._target_sparsity = config.get("target_sparsity", 0.5)
        self._prefill_ratio = config.get("prefill_ratio", 1)
        self._sparse_ratio = config.get("sparse_ratio", 1.0)

        # Create a ModelConfig for lighteval pipeline
        self.config = ModelConfig(model_name=self._model_name)

        # Create args object for setup_model
        args = argparse.Namespace(
            model_name=self._model_name,
            method=self._method,
            config_file=self._config_file,
            sparse_config_file=self._sparse_config_file,
            cache_dir=self._cache_dir,
            device=self._device,
            target_sparsity=self._target_sparsity,
            prefill_ratio=self._prefill_ratio,
            sparse_ratio=self._sparse_ratio,
        )

        # Load model using existing setup
        self._model_config, self._tokenizer, self._model = setup_model(args)
        self._model = self._model.eval()

        # Set padding token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Initialize cache for lighteval pipeline
        self._cache = SampleCache(self.config)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self):
        return self._model.config.max_position_embeddings

    def greedy_until(self, docs, override_bs=None):
        """Generate text until stop sequence or max tokens."""
        results = []
        for doc in docs:
            context = doc.query
            stop_sequences = doc.stop_sequences
            max_tokens = doc.generation_size or 256

            inputs = self._tokenizer(context, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            generated_text = self._tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[
                            : generated_text.index(stop_seq)
                        ]

            generated_tokens = outputs[0][input_ids.shape[1] :].tolist()
            results.append(
                ModelResponse(
                    text=[generated_text],
                    input_tokens=input_ids[0].tolist(),
                    output_tokens=[generated_tokens],
                )
            )

        return results

    def loglikelihood(self, docs, override_bs=None):
        """Compute log probabilities of continuations."""
        results = []
        for doc in docs:
            context = doc.query
            # For loglikelihood, we compute for each choice
            choices = doc.choices if doc.choices else [""]

            doc_results = []
            for continuation in choices:
                # Tokenize context and continuation
                context_ids = self._tokenizer.encode(context, add_special_tokens=False)
                continuation_ids = self._tokenizer.encode(
                    continuation, add_special_tokens=False
                )
                full_ids = context_ids + continuation_ids

                input_ids = torch.tensor([full_ids], device=self._device)

                with torch.no_grad():
                    outputs = self._model(input_ids=input_ids)
                    logits = outputs.logits

                # Get log probs for continuation tokens
                shift_logits = logits[0, len(context_ids) - 1 : -1, :]
                shift_labels = torch.tensor(continuation_ids, device=self._device)

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs[range(len(continuation_ids)), shift_labels]

                total_log_prob = token_log_probs.sum().item()
                is_greedy = all(
                    shift_logits[i].argmax().item() == shift_labels[i].item()
                    for i in range(len(continuation_ids))
                )

                doc_results.append((total_log_prob, is_greedy, continuation_ids))

            # Return results for all choices
            results.append(
                ModelResponse(
                    logprobs=[r[0] for r in doc_results],
                    argmax_logits_eq_gold=[r[1] for r in doc_results],
                    input_tokens=context_ids,
                    output_tokens=[r[2] for r in doc_results],
                )
            )

        return results

    def loglikelihood_rolling(self, docs, override_bs=None):
        """Compute rolling log probabilities."""
        results = []
        for doc in docs:
            context = doc.query

            input_ids = self._tokenizer.encode(context, return_tensors="pt").to(
                self._device
            )

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids)
                logits = outputs.logits

            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

            token_log_probs_list = token_log_probs.tolist()
            input_tokens = input_ids[0].tolist()

            results.append(
                ModelResponse(
                    logprobs=token_log_probs_list,
                    input_tokens=input_tokens,
                    output_tokens=[[t] for t in input_tokens[1:]],
                )
            )

        return results

    def loglikelihood_single_token(self, docs, override_bs=None):
        """Compute log probabilities for single token predictions."""
        results = []
        for doc in docs:
            context = doc.query
            choices = doc.choices  # List of single tokens

            input_ids = self._tokenizer.encode(context, return_tensors="pt").to(
                self._device
            )

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids)
                logits = outputs.logits

            # Get logits for the last position
            last_logits = logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

            choice_log_probs = []
            choice_token_ids = []
            for choice in choices:
                choice_id = self._tokenizer.encode(choice, add_special_tokens=False)
                if len(choice_id) > 0:
                    choice_log_probs.append(log_probs[choice_id[0]].item())
                    choice_token_ids.append([choice_id[0]])
                else:
                    choice_log_probs.append(float("-inf"))
                    choice_token_ids.append([])

            # Determine if the argmax matches each choice
            argmax_token = last_logits.argmax().item()
            argmax_eq_gold = [
                len(cid) > 0 and cid[0] == argmax_token for cid in choice_token_ids
            ]

            results.append(
                ModelResponse(
                    logprobs=choice_log_probs,
                    argmax_logits_eq_gold=argmax_eq_gold,
                    input_tokens=input_ids[0].tolist(),
                    output_tokens=choice_token_ids,
                )
            )

        return results


def parse_args():
    parser = argparse.ArgumentParser()
    # Model setup
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "relufiction", "r_sparse"],
    )
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument("--prefill_ratio", type=float, default=0.1)
    parser.add_argument("--sparse_ratio", type=float, default=1)
    parser.add_argument(
        "--config_file", type=str, default="config/llama-3-8b_default.json"
    )
    parser.add_argument("--sparse_config_file", type=str, default=None)

    # Evaluation setup
    # Task format: suite|task|num_few_shot
    parser.add_argument("--tasks", type=str, default="leaderboard|mmlu|0")
    parser.add_argument("--output_dir", type=str, default="./evals")
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Create model instance directly
    model_config = {
        "model_name": args.model_name,
        "method": args.method,
        "config_file": args.config_file,
        "sparse_config_file": args.sparse_config_file,
        "cache_dir": args.cache_dir,
        "device": args.device,
        "target_sparsity": args.target_sparsity,
        "prefill_ratio": args.prefill_ratio,
        "sparse_ratio": args.sparse_ratio,
    }

    model = RSparseModel(model_config)

    # Set up evaluation tracking
    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir, save_details=True
    )

    # Configure pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
        # override_batch_size=args.batch_size,
    )

    # Create and run the pipeline
    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
