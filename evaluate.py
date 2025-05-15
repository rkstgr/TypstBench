import os
import json
from pathlib import Path
import time
import argparse
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple
import uuid
import litellm
from litellm.types.utils import ModelResponse, Choices
from dataset import EvaluationMethod, TypstBenchDataset, TypstBenchSample
from typst import TypstRenderResult, TypstRenderer
from typst import get_typst_code

class EvaluateResult:
    """Result of evaluating a model's output against a target output."""

    def __init__(self,
                 model_render_result: TypstRenderResult,
                 target_render_result: Optional[TypstRenderResult] = None,
                 is_correct: bool = False,
                 error_message: Optional[str] = None,
                 comparison_method: str = "none"):
        """
        Initialize the evaluation result.

        Args:
            model_render_result: Result of rendering the model's output
            target_render_result: Result of rendering the target output
            is_correct: Whether the model output matches the target output
            error_message: Error message if something went wrong
            comparison_method: Method used for comparison
        """
        self.model_render_result = model_render_result
        self.target_render_result = target_render_result
        self.is_correct = is_correct
        self.error_message = error_message
        self.comparison_method = comparison_method

    def __str__(self) -> str:
        if self.is_correct:
            return f"Correct (compared using {self.comparison_method})"
        elif self.error_message:
            return f"Evaluation error: {self.error_message}"
        else:
            return f"Incorrect (compared using {self.comparison_method})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "is_correct": self.is_correct,
            "comparison_method": self.comparison_method,
            "error_message": self.error_message,
            "model_compile_success": self.model_render_result.success if self.model_render_result else False,
            "target_compile_success": self.target_render_result.success if self.target_render_result else False,
            "model_errors": [str(e) for e in self.model_render_result.errors] if self.model_render_result and self.model_render_result.errors else []
        }

    def cleanup(self) -> None:
        """Clean up any temporary files created during evaluation."""
        if self.model_render_result:
            self.model_render_result.cleanup()
        if self.target_render_result:
            self.target_render_result.cleanup()


def calculate_file_md5(filepath: str) -> Optional[str]:
    """Calculate MD5 hash of a file."""
    if not os.path.exists(filepath):
        return None

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def evaluate_output(model_output: str, 
                   target_output: str,
                   renderer: Optional[TypstRenderer] = None,
                   save_dir: Optional[str] = None,
                   task_path: Optional[str] = None) -> EvaluateResult:
    """
    Evaluates if model_output and target_output match.
    Saves PDFs to save_dir if specified for later inspection.
    """
    if not renderer:
        renderer = TypstRenderer()
    
    # Generate unique identifier for this evaluation
    eval_id = task_path or str(uuid.uuid4())[:8]
    
    # Create persistent paths if save_dir is provided
    persistent_paths = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        persistent_paths = {
            'model': os.path.join(save_dir, f"{eval_id}_model.pdf"),
            'target': os.path.join(save_dir, f"{eval_id}_target.pdf")
        }
    
    # Render model output
    model_render_result = renderer.render(get_typst_code(model_output), pdf_path=persistent_paths["model"])
    if not model_render_result.success:
        return EvaluateResult(
            model_render_result=model_render_result,
            is_correct=False,
            comparison_method="compile_check"
        )
    
    # Render target output
    target_render_result = renderer.render(get_typst_code(target_output), pdf_path=persistent_paths["target"])
    if not target_render_result.success:
        return EvaluateResult(
            model_render_result=model_render_result,
            target_render_result=target_render_result,
            is_correct=False,
            error_message="Target output failed to render - dataset error",
            comparison_method="compile_check"
        )
    # Compare PDF hashes (using original paths for comparison)
    model_hash = calculate_file_md5(model_render_result.pdf_path)
    target_hash = calculate_file_md5(target_render_result.pdf_path)
    
    if not model_hash or not target_hash:
        return EvaluateResult(
            model_render_result=model_render_result,
            target_render_result=target_render_result,
            is_correct=False,
            error_message="Failed to calculate PDF hash",
            comparison_method="pdf_hash"
        )
    
    # Compare hashes
    is_correct = model_hash == target_hash
    
    return EvaluateResult(
        model_render_result=model_render_result,
        target_render_result=target_render_result,
        is_correct=is_correct,
        comparison_method="pdf_hash"
    )


class TypstBenchEvaluator:
    """Evaluates LLM performance on the TypstBench dataset."""

    def __init__(
        self,
        dataset: TypstBenchDataset,
        model: str,
        api_key: Optional[str] = None,
        output_dir: str = "results",
        artifacts_dir: str = "pdf_artifacts",
        prompt_template: str = "Task type: {task_type}\nTask:\n{input}",
        max_concurrent_requests: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: The TypstBench dataset
            model: The LLM model identifier to use with LiteLLM
            api_key: API key for the model provider (if required)
            output_dir: Directory to save evaluation results
            prompt_template: Template string for formatting prompts
        """
        self.dataset = dataset
        self.model = model
        self.output_dir = output_dir
        self.prompt_template = prompt_template
        self.artifacts_dir = artifacts_dir
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Set up API key if provided
        if api_key:
            if "openai" in model.lower():
                os.environ["OPENAI_API_KEY"] = api_key
            elif "anthropic" in model.lower():
                os.environ["ANTHROPIC_API_KEY"] = api_key
            else:
                # For other providers, let litellm handle it
                litellm.api_key = api_key

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

    def _format_prompt(self, sample: TypstBenchSample) -> str:
        """Format the prompt for a sample using the prompt template."""
        return self.prompt_template.format(task_type=sample.category, input=sample.raw_input)

    async def _evaluate_sample(self, sample: TypstBenchSample) -> Dict[str, Any]:
        """
        Evaluate a single sample using the LLM.

        Args:
            sample: The sample to evaluate

        Returns:
            A dictionary with evaluation results
        """
        async with self.semaphore:
            prompt = self._format_prompt(sample)
            start_time = time.time()

            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        {
                        "role": "system", "content": """
This is the TypstBench evaluation system. You are being tested on your ability to generate typst code and knowledge of typst.
You are given two types of tasks:

- generate: Generate typst code based on the input.
Your answer should be a single typst code block.
Example answer:
```typst
<typst code here>
```

- multiple_choice: Choose the correct answer from the options provided.
There will alwyas be four options: A, B, C, D, and either one or two of them will be correct.
If only one is correct, answer with the letter of the correct option.
Example answer:
A

If two are correct, answer with the letters of the correct options separated by a comma.
Example answer:
A,B
"""
                        },
                        {
                            "role": "user", "content": prompt
                        }],
                    temperature=0.1,  # Low temperature for more consistent outputs
                    max_tokens=1024,
                )

                assert isinstance(response, ModelResponse)
                assert isinstance(response.choices[0], Choices)

                content = response.choices[0].message.content
                assert content

                model_output = content.strip()
                
                if sample.evaluation_method == EvaluationMethod.PDF_HASH:
                    sample_artifacts_dir = Path(sample.get_path("pdf_artifacts")).parent.absolute()
                    eval_result = evaluate_output(
                        model_output, 
                        sample.raw_ground_truth,
                        save_dir=sample_artifacts_dir,
                        task_path=sample.file_path
                    )
                else:
                    is_correct = model_output == sample.raw_ground_truth
                    eval_result = EvaluateResult(
                        model_render_result=None,
                        target_render_result=None,
                        is_correct=is_correct,
                        comparison_method="string_match"
                    )

                result = {
                    "task": sample.file_path,
                    "prompt": prompt,
                    "expected_output": sample.raw_ground_truth,
                    "model_output": model_output,
                    "is_correct": eval_result.is_correct,
                    "latency_seconds": time.time() - start_time,
                    "model": self.model,
                    "metadata": sample.metadata,
                    "eval_result": eval_result.to_dict(),
                    "model_pdf_path": getattr(eval_result.model_render_result, "persistent_pdf_path", None),
                    "target_pdf_path": getattr(eval_result.target_render_result, "persistent_pdf_path", None),
                }

                return result

            except Exception as e:
                return {
                    "task": sample.file_path,
                    "prompt": prompt,
                    "expected_output": sample.raw_ground_truth,
                    "model_output": None,
                    "is_correct": False,
                    "error": str(e),
                    "latency_seconds": time.time() - start_time,
                    "model": self.model,
                    "metadata": sample.metadata,
                }

    async def evaluate_samples(self, samples: List[TypstBenchSample]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple samples in parallel.

        Args:
            samples: List of samples to evaluate

        Returns:
            List of evaluation results
        """
        tasks = [self._evaluate_sample(sample) for sample in samples]
        return await asyncio.gather(*tasks)

    async def evaluate_dataset(
        self,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate samples from the dataset with optional filtering.

        Args:
            filter_kwargs: Arguments to filter samples (passed to dataset.filter_samples)
            max_samples: Maximum number of samples to evaluate

        Returns:
            A tuple containing (summary_metrics, individual_results)
        """
        # Filter samples if needed
        if filter_kwargs:
            samples = self.dataset.filter_samples(**filter_kwargs)
        else:
            samples = self.dataset.get_all_samples()

        # Limit the number of samples if specified
        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]

        print(f"Evaluating {len(samples)} samples with model: {self.model}")

        # Evaluate all samples
        results = await self.evaluate_samples(samples)

        # Calculate summary metrics
        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0

        avg_latency = sum(r["latency_seconds"] for r in results if "latency_seconds" in r) / total

        # Calculate tier-specific metrics if available
        tier_metrics = {}
        tiers = set(r["metadata"].get("tier") for r in results if "tier" in r["metadata"])

        for tier in tiers:
            tier_results = [r for r in results if r["metadata"].get("tier") == tier]
            tier_total = len(tier_results)
            tier_correct = sum(1 for r in tier_results if r["is_correct"])
            tier_accuracy = tier_correct / tier_total if tier_total > 0 else 0

            tier_metrics[tier] = {
                "total": tier_total,
                "correct": tier_correct,
                "accuracy": tier_accuracy
            }

        # Calculate feature-specific metrics
        feature_metrics = {}
        all_features = set()
        for r in results:
            features = r["metadata"].get("features", [])
            if features:
                all_features.update(features)

        for feature in all_features:
            feature_results = [r for r in results if feature in r["metadata"].get("features", [])]
            feature_total = len(feature_results)
            feature_correct = sum(1 for r in feature_results if r["is_correct"])
            feature_accuracy = feature_correct / feature_total if feature_total > 0 else 0

            feature_metrics[feature] = {
                "total": feature_total,
                "correct": feature_correct,
                "accuracy": feature_accuracy
            }

        # Compile summary metrics
        summary = {
            "model": self.model,
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": accuracy,
            "average_latency_seconds": avg_latency,
            "tier_metrics": tier_metrics,
            "feature_metrics": feature_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return summary, results

    async def run_evaluation(
        self,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None
    ) -> str:
        """
        Run evaluation and save results to file.

        Args:
            filter_kwargs: Arguments to filter samples
            max_samples: Maximum number of samples to evaluate

        Returns:
            Path to the saved results file
        """
        summary, results = await self.evaluate_dataset(filter_kwargs, max_samples)

        # Create result object with both summary and detailed results
        evaluation_result = {
            "summary": summary,
            "detailed_results": results
        }

        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = self.model.replace("/", "-").replace(":", "-")
        filename = f"typstbench_{model_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, indent=2)

        print(f"Results saved to: {filepath}")

        # Print summary to console
        print("\nEvaluation Summary:")
        print(f"Model: {summary['model']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2%} ({summary['correct_samples']}/{summary['total_samples']})")
        print(f"Average Latency: {summary['average_latency_seconds']:.2f} seconds")

        if summary['tier_metrics']:
            print("\nAccuracy by Tier:")
            for tier, metrics in summary['tier_metrics'].items():
                print(f"  {tier}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        return filepath


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on the TypstBench dataset")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model identifier for litellm")
    parser.add_argument("--api-key", help="API key for the model provider")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing the dataset")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--category", help="Filter samples by category (e.g., 'multiple_choice')")
    parser.add_argument("--tier", help="Filter samples by tier (e.g., 'basic')")
    parser.add_argument("--features", help="Filter samples by features (comma-separated)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests")
    

    args = parser.parse_args()

    # Load dataset
    dataset = TypstBenchDataset(args.dataset_dir)

    # Prepare filter arguments
    filter_kwargs = {}
    if args.category:
        filter_kwargs["category"] = args.category
    if args.tier:
        filter_kwargs["tier"] = args.tier
    if args.features:
        filter_kwargs["features"] = args.features.split(",")

    # Initialize evaluator
    evaluator = TypstBenchEvaluator(
        dataset=dataset,
        model=args.model,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_concurrent_requests=args.concurrency,
    )

    # Run evaluation
    await evaluator.run_evaluation(
        filter_kwargs=filter_kwargs if filter_kwargs else None,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    asyncio.run(main())
