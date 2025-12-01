"""
Custom model wrapper for R-Sparse evaluation with lighteval 0.10.0
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List, Union

import torch
from transformers import AutoTokenizer, AutoConfig

from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_output import (
    GenerativeResponse,
    LoglikelihoodResponse,
    LoglikelihoodSingleTokenResponse,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

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
        self._prefill_ratio = config.get("prefill_ratio", 0.1)
        self._sparse_ratio = config.get("sparse_ratio", 1.0)

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

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_info(self):
        return ModelInfo(
            model_name=self._model_name,
            model_sha="",
            model_dtype=str(self._model.dtype)
            if hasattr(self._model, "dtype")
            else "float16",
            model_size="",
        )

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

    def greedy_until(self, requests, override_bs=None):
        """Generate text until stop sequence or max tokens.

        Returns list of GenerativeResponse objects.
        """
        results = []
        for request in requests:
            # v0.10.0 uses request objects with tokenized_context
            tokenized_context = getattr(request, "tokenized_context", None)
            if tokenized_context is not None:
                input_ids = torch.tensor([list(tokenized_context)], device=self._device)
            else:
                context = getattr(request, "context", "") or ""
                input_ids = self._tokenizer.encode(context, return_tensors="pt").to(
                    self._device
                )

            stop_sequences = getattr(request, "stop_sequence", None)
            max_tokens = getattr(request, "generation_size", 256) or 256

            attention_mask = torch.ones_like(input_ids)

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
                if isinstance(stop_sequences, str):
                    stop_sequences = [stop_sequences]
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[
                            : generated_text.index(stop_seq)
                        ]

            generated_tokens = outputs[0][input_ids.shape[1] :].tolist()

            results.append(
                GenerativeResponse(
                    result=generated_text,
                    input_tokens=input_ids[0].tolist(),
                    generated_tokens=generated_tokens,
                )
            )

        return results

    def loglikelihood(self, requests, override_bs=None):
        """Compute log probabilities of continuations.

        Returns list of LoglikelihoodResponse objects.
        Each response contains (logprob, is_greedy) tuple.
        """
        results = []
        for request in requests:
            # v0.10.0 provides tokenized context and continuation
            context_ids = getattr(request, "tokenized_context", None)
            continuation_ids = getattr(request, "tokenized_continuation", None)

            # Convert to list if not None
            if context_ids is not None:
                context_ids = list(context_ids)
            else:
                context = getattr(request, "context", "") or ""
                context_ids = self._tokenizer.encode(context, add_special_tokens=False)

            if continuation_ids is not None:
                continuation_ids = list(continuation_ids)
            else:
                continuation = getattr(request, "choice", "") or ""
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

            results.append(
                LoglikelihoodResponse(
                    result=(total_log_prob, is_greedy),
                    input_tokens=context_ids,
                    generated_tokens=continuation_ids,
                )
            )

        return results

    def loglikelihood_rolling(self, requests, override_bs=None):
        """Compute rolling log probabilities for perplexity.

        Returns list of LoglikelihoodResponse objects.
        """
        results = []
        for request in requests:
            tokenized_context = getattr(request, "tokenized_context", None)
            if tokenized_context is not None:
                input_ids = torch.tensor([list(tokenized_context)], device=self._device)
            else:
                context = getattr(request, "context", "") or ""
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

            total_log_prob = token_log_probs.sum().item()
            input_tokens = input_ids[0].tolist()

            results.append(
                LoglikelihoodResponse(
                    result=(total_log_prob, False),
                    input_tokens=input_tokens,
                    generated_tokens=input_tokens[1:],
                )
            )

        return results

    def loglikelihood_single_token(self, requests, override_bs=None):
        """Compute log probabilities for single token predictions.

        Returns list of LoglikelihoodSingleTokenResponse objects.
        Each response contains list of log probs for each choice.
        """
        results = []
        for request in requests:
            tokenized_context = getattr(request, "tokenized_context", None)
            if tokenized_context is not None:
                input_ids = torch.tensor([list(tokenized_context)], device=self._device)
            else:
                context = getattr(request, "context", "") or ""
                input_ids = self._tokenizer.encode(context, return_tensors="pt").to(
                    self._device
                )

            choices = getattr(request, "choices", []) or []

            with torch.no_grad():
                outputs = self._model(input_ids=input_ids)
                logits = outputs.logits

            # Get logits for the last position
            last_logits = logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

            choice_log_probs = []
            for choice in choices:
                if (
                    hasattr(request, "tokenized_continuation")
                    and request.tokenized_continuation
                ):
                    choice_id = request.tokenized_continuation
                else:
                    choice_id = self._tokenizer.encode(choice, add_special_tokens=False)

                if len(choice_id) > 0:
                    choice_log_probs.append(log_probs[choice_id[0]].item())
                else:
                    choice_log_probs.append(float("-inf"))

            results.append(
                LoglikelihoodSingleTokenResponse(
                    result=choice_log_probs,
                    input_tokens=input_ids[0].tolist(),
                    generated_tokens=[],
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
    # Task format for v0.10.0: suite|task|num_few_shot|truncate_flag
    parser.add_argument("--tasks", type=str, default="lighteval|arc:easy|0|0")
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
