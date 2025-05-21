import os
import json
from pathlib import Path
import time
import argparse
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, TypedDict
import litellm
from litellm.types.utils import ModelResponse, Choices
from dataset import ComparisonMethod, TypstBenchDataset, TypstBenchSample
from typst import TypstRenderer, get_typst_code

class EvaluateTaskResult(TypedDict):
    task: str
    prompt: str
    expected_output: str
    model_output: str
    is_correct: bool
    latency_seconds: float
    model: str
    metadata: Dict[str, Any]
    comparison_result: Dict[str, Any]

class EvaluationRunSummary(TypedDict):
    model: str
    total_samples: int
    correct_samples: int
    overall_accuracy: float
    average_latency_seconds: float
    tier_metrics: Dict[str, Dict[str, Any]]
    feature_metrics: Dict[str, Dict[str, Any]]
    timestamp: str
    system_prompt: str
    temperature: float

class EvaluationRunResults(TypedDict):
    summary: EvaluationRunSummary
    results_per_task: List[EvaluateTaskResult]

class ComparisonResult(TypedDict):
    is_correct: bool
    error_message: Optional[str]
    comparison_method: str


def calculate_file_md5(filepath: str) -> Optional[str]:
    """Calculate MD5 hash of a file."""
    if not os.path.exists(filepath):
        return None

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_outputs(model_output: str,
                    target_output: str,
                    comparison_method: ComparisonMethod,
                    debug_artificts_dir: Optional[Path] = None
                    ) -> ComparisonResult:
    if comparison_method == ComparisonMethod.STRING_MATCH:
        is_correct = model_output.strip() == target_output.strip()
        return ComparisonResult(
            is_correct=is_correct,
            error_message=None,
            comparison_method="string_match"
        )
    
    elif comparison_method == ComparisonMethod.PDF_HASH:
        model_typst_code = get_typst_code(model_output)
        if not model_typst_code:
            return ComparisonResult(
                is_correct=False,
                error_message="Model output is not valid Typst code",
                comparison_method="compile_check"
            )
        model_render_result = TypstRenderer().render(model_typst_code)
        if not model_render_result["success"]:
            return ComparisonResult(
                is_correct=False,
                error_message=model_render_result["error_output"],
                comparison_method="compile_check"
            )
        target_typst_code = get_typst_code(target_output)
        if not target_typst_code:
            return ComparisonResult(
                is_correct=False,
                error_message="Target output is not valid Typst code",
                comparison_method="compile_check"
            )
        target_render_result = TypstRenderer().render(target_typst_code)
        if not target_render_result["success"]:
            return ComparisonResult(
                is_correct=False,
                error_message="Target output failed to render - dataset error: " + target_render_result["error_output"],
                comparison_method="compile_check"
            )

        model_hash = calculate_file_md5(model_render_result["pdf_path"])
        target_hash = calculate_file_md5(target_render_result["pdf_path"])
        if not model_hash or not target_hash:
            return ComparisonResult(
                is_correct=False,
                error_message="Failed to calculate PDF hash",
                comparison_method="pdf_hash"
            )
        
        is_correct = model_hash == target_hash
        if not is_correct and debug_artificts_dir:
            pdf_path = Path(model_render_result["pdf_path"])
            os.makedirs(debug_artificts_dir, exist_ok=True)
            pdf_path.rename(debug_artificts_dir / "model.pdf")

            return ComparisonResult(
                is_correct=False,
                error_message="PDF hashes do not match. Model output saved to: " + str(debug_artificts_dir / "model.pdf"),
                comparison_method="pdf_hash"
            )
            

        return ComparisonResult(
            is_correct=is_correct,
            error_message=None,
            comparison_method="pdf_hash"
        )
    
    else:
        raise ValueError(f"Unknown evaluation method: {comparison_method}")


class TypstBenchEvaluator:
    """Evaluates LLM performance on the TypstBench dataset."""

    def __init__(
        self,
        dataset: TypstBenchDataset,
        model: str,
        api_key: Optional[str] = None,
        artifacts_dir: str = "pdf_artifacts",
        prompt_template: str = "Task type: {task_type}\nTask:\n{input}",
        max_concurrent_requests: int = 5
    ):
        """
        Initialize the evaluator.
        """

        self.dataset = dataset
        self.model = model
        self.prompt_template = prompt_template
        self.artifacts_dir = artifacts_dir
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)


        # MODEL SETTINGS
        self.TEMPERATURE = 0.1
        self.SYSTEM_PROMPT = """
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
        os.makedirs(artifacts_dir, exist_ok=True)

    def _format_prompt(self, sample: TypstBenchSample) -> str:
        """Format the prompt for a sample using the prompt template."""
        return self.prompt_template.format(task_type=sample.category, input=sample.raw_input)

    async def _evaluate_sample(self, sample: TypstBenchSample) -> EvaluateTaskResult:
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
                        { "role": "system", "content": self.SYSTEM_PROMPT},
                        { "role": "user", "content": prompt}
                        ],
                    temperature=self.TEMPERATURE,  # Low temperature for more consistent outputs
                    max_tokens=1024,
                )

                assert isinstance(response, ModelResponse)
                assert isinstance(response.choices[0], Choices)

                content = response.choices[0].message.content
                assert content

                model_output = content.strip()

                comparison_result = compare_outputs(
                    model_output,
                    sample.raw_ground_truth,
                    sample.evaluation_method,
                    debug_artificts_dir=Path(self.artifacts_dir) / sample.category / sample.task_number
                )

                return EvaluateTaskResult(
                    task=sample.file_path,
                    prompt=prompt,
                    expected_output=sample.raw_ground_truth,
                    model_output=model_output,
                    is_correct=comparison_result["is_correct"],
                    latency_seconds=time.time() - start_time,
                    model=self.model,
                    metadata=sample.metadata,
                    comparison_result=comparison_result,
                )

            except Exception as e:
                return EvaluateTaskResult(
                    task=sample.file_path,
                    prompt=prompt,
                    expected_output=sample.raw_ground_truth,
                    model_output="",
                    is_correct=False,
                    latency_seconds=time.time() - start_time,
                    model=self.model,
                    metadata=sample.metadata,
                    comparison_result={
                        "is_correct": False,
                        "error_message": str(e),
                        "comparison_method": "error"
                    }
                )

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
    ) -> EvaluationRunResults:
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
            "system_prompt": self.SYSTEM_PROMPT,
            "temperature": self.TEMPERATURE,
        }

        return EvaluationRunResults(
            summary=summary,
            results_per_task=results,
        )

    async def run_evaluation(
        self,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None
    ) -> EvaluationRunResults:
        
        return await self.evaluate_dataset(filter_kwargs, max_samples)


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
    parser.add_argument("--task", help="Path to a specific task to evaluate")
    

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
    if args.task:
        category, task_number = args.task.split("/")
        filter_kwargs["category"] = category
        filter_kwargs["task_number"] = task_number

    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    model_name = args.model.replace("/", "_").replace(":", "_")
    model_timestamp = f"{model_name}-{timestamp}"
    results_dir = Path(args.output_dir) / model_timestamp
    os.makedirs(results_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = TypstBenchEvaluator(
        dataset=dataset,
        model=args.model,
        api_key=args.api_key,
        max_concurrent_requests=args.concurrency,
        artifacts_dir=results_dir,
    )

    # Run evaluation
    eval_results = await evaluator.run_evaluation(
        filter_kwargs=filter_kwargs if filter_kwargs else None,
        max_samples=args.max_samples
    )

    results_file = results_dir / "results.json"
    

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    summary = eval_results["summary"]
    # Print summary to console
    print("\nEvaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%} ({summary['correct_samples']}/{summary['total_samples']})")
    print(f"Average Latency: {summary['average_latency_seconds']:.2f} seconds")

    if summary['tier_metrics']:
        print("\nAccuracy by Tier:")
        for tier, metrics in summary['tier_metrics'].items():
            print(f"  {tier}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    asyncio.run(main())
