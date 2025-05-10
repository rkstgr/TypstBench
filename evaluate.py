import os
import json
import time
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import litellm
from litellm.types.utils import ModelResponse, Choices
from dataset import TypstBenchDataset, TypstBenchSample

class TypstBenchEvaluator:
    """Evaluates LLM performance on the TypstBench dataset."""

    def __init__(
        self,
        dataset: TypstBenchDataset,
        model: str,
        api_key: Optional[str] = None,
        output_dir: str = "results",
        prompt_template: str = "Convert this description to Typst code:\n\n{input}\n\nTypst code:"
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

    def _format_prompt(self, sample: TypstBenchSample) -> str:
        """Format the prompt for a sample using the prompt template."""
        return self.prompt_template.format(input=sample.raw_input)

    async def _evaluate_sample(self, sample: TypstBenchSample) -> Dict[str, Any]:
        """
        Evaluate a single sample using the LLM.

        Args:
            sample: The sample to evaluate

        Returns:
            A dictionary with evaluation results
        """
        prompt = self._format_prompt(sample)
        start_time = time.time()

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for more consistent outputs
                max_tokens=1024,
            )

            assert isinstance(response, ModelResponse)
            assert isinstance(response.choices[0], Choices)

            content = response.choices[0].message.content
            assert content

            completion_text = content.strip()

            # Perform exact string matching evaluation
            is_correct = completion_text == sample.raw_output

            result = {
                "sample_id": sample.id,
                "prompt": prompt,
                "expected_output": sample.raw_output,
                "model_output": completion_text,
                "is_correct": is_correct,
                "latency_seconds": time.time() - start_time,
                "model": self.model,
                "metadata": sample.metadata,
            }

            return result

        except Exception as e:
            return {
                "sample_id": sample.id,
                "prompt": prompt,
                "expected_output": sample.raw_output,
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
    parser.add_argument("--tier", help="Filter samples by tier (e.g., 'basic')")
    parser.add_argument("--features", help="Filter samples by features (comma-separated)")
    parser.add_argument("--difficulty", type=int, help="Filter samples by difficulty level")
    parser.add_argument("--task-type", help="Filter samples by task type")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")

    args = parser.parse_args()

    # Load dataset
    dataset = TypstBenchDataset(args.dataset_dir)

    # Prepare filter arguments
    filter_kwargs = {}
    if args.tier:
        filter_kwargs["tier"] = args.tier
    if args.features:
        filter_kwargs["features"] = args.features.split(",")
    if args.difficulty is not None:
        filter_kwargs["difficulty"] = args.difficulty
    if args.task_type:
        filter_kwargs["task_type"] = args.task_type

    # Initialize evaluator
    evaluator = TypstBenchEvaluator(
        dataset=dataset,
        model=args.model,
        api_key=args.api_key,
        output_dir=args.output_dir
    )

    # Run evaluation
    await evaluator.run_evaluation(
        filter_kwargs=filter_kwargs if filter_kwargs else None,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    asyncio.run(main())
