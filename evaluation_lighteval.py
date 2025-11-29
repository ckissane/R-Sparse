"""
Custom model wrapper for R-Sparse evaluation with lighteval 0.10.0
"""

import argparse
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import (
    ModelReturn,
    GenerateReturn,
    LoglikelihoodReturn,
    LoglikelihoodSingleTokenReturn,
)
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager

import torch
from transformers import AutoTokenizer, AutoConfig
from utils.setup import setup_model, setup_config


class RSparseModel(LightevalModel):
    """Custom lighteval model wrapper for R-Sparse models."""

    def __init__(self, config, env_config):
        self.model_name = config.get("model_name", "meta-llama/Meta-Llama-3-8B")
        self.method = config.get("method", "full")
        self.config_file = config.get("config_file", None)
        self.sparse_config_file = config.get("sparse_config_file", None)
        self.cache_dir = config.get("cache_dir", None)
        self.device = config.get("device", "cuda:0")
        self.target_sparsity = config.get("target_sparsity", 0.5)
        self.prefill_ratio = config.get("prefill_ratio", 0.1)
        self.sparse_ratio = config.get("sparse_ratio", 1.0)

        # Create args object for setup_model
        self.args = argparse.Namespace(
            model_name=self.model_name,
            method=self.method,
            config_file=self.config_file,
            sparse_config_file=self.sparse_config_file,
            cache_dir=self.cache_dir,
            device=self.device,
            target_sparsity=self.target_sparsity,
            prefill_ratio=self.prefill_ratio,
            sparse_ratio=self.sparse_ratio,
        )

        # Load model using existing setup
        self._config, self.tokenizer, self.model = setup_model(self.args)
        self.model = self.model.eval().to(self.device)

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def add_special_tokens(self):
        return False

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    def greedy_until(self, requests, override_bs=None):
        """Generate text until stop sequence or max tokens."""
        results = []
        for request in requests:
            context = request.context
            stop_sequences = request.stop_sequence
            max_tokens = request.generation_size or 256

            inputs = self.tokenizer(context, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1] :], skip_special_tokens=True
            )

            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[
                            : generated_text.index(stop_seq)
                        ]

            results.append(
                GenerateReturn(
                    result=generated_text,
                    logits=None,
                    generated_tokens=[],
                    input_tokens=[],
                )
            )

        return results

    def loglikelihood(self, requests, override_bs=None):
        """Compute log probabilities of continuations."""
        results = []
        for request in requests:
            context = request.context
            continuation = request.choice

            # Tokenize context and continuation
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            continuation_ids = self.tokenizer.encode(
                continuation, add_special_tokens=False
            )
            full_ids = context_ids + continuation_ids

            input_ids = torch.tensor([full_ids], device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

            # Get log probs for continuation tokens
            shift_logits = logits[0, len(context_ids) - 1 : -1, :]
            shift_labels = torch.tensor(continuation_ids, device=self.device)

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[range(len(continuation_ids)), shift_labels]

            total_log_prob = token_log_probs.sum().item()
            is_greedy = all(
                shift_logits[i].argmax().item() == shift_labels[i].item()
                for i in range(len(continuation_ids))
            )

            results.append(
                LoglikelihoodReturn(
                    result=(total_log_prob, is_greedy),
                    input_tokens=[],
                    generated_tokens=[],
                    truncated=[],
                )
            )

        return results

    def loglikelihood_rolling(self, requests, override_bs=None):
        """Compute rolling log probabilities."""
        results = []
        for request in requests:
            context = request.context

            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

            shift_logits = logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

            total_log_prob = token_log_probs.sum().item()

            results.append(
                LoglikelihoodReturn(
                    result=(total_log_prob, False),
                    input_tokens=[],
                    generated_tokens=[],
                    truncated=[],
                )
            )

        return results

    def loglikelihood_single_token(self, requests, override_bs=None):
        """Compute log probabilities for single token predictions."""
        results = []
        for request in requests:
            context = request.context
            choices = request.choices  # List of single tokens

            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

            # Get logits for the last position
            last_logits = logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

            choice_log_probs = []
            for choice in choices:
                choice_id = self.tokenizer.encode(choice, add_special_tokens=False)
                if len(choice_id) > 0:
                    choice_log_probs.append(log_probs[choice_id[0]].item())
                else:
                    choice_log_probs.append(float("-inf"))

            results.append(
                LoglikelihoodSingleTokenReturn(
                    result=choice_log_probs,
                    input_tokens=[],
                    generated_tokens=[],
                    truncated=[],
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
    parser.add_argument("--tasks", type=str, default="leaderboard|mmlu|5|0")
    parser.add_argument("--output_dir", type=str, default="./evals")
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up evaluation tracking
    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir, save_details=True
    )

    # Configure pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.CUSTOM,
        override_batch_size=args.batch_size,
    )

    # Configure custom model
    model_config = CustomModelConfig(
        model="r-sparse-custom",
        model_definition_file_path=__file__,
        model_class="RSparseModel",
        model_args={
            "model_name": args.model_name,
            "method": args.method,
            "config_file": args.config_file,
            "sparse_config_file": args.sparse_config_file,
            "cache_dir": args.cache_dir,
            "device": args.device,
            "target_sparsity": args.target_sparsity,
            "prefill_ratio": args.prefill_ratio,
            "sparse_ratio": args.sparse_ratio,
        },
    )

    # Create and run the pipeline
    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
