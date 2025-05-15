#!/usr/bin/env python3
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import time

# Import the evaluator components from the existing script
# Assuming they are in the same directory or available as imports
from evaluate import TypstBenchEvaluator, TypstBenchDataset


class BenchmarkRunner:
    def __init__(self, config_path: str = "eval.config.json", dataset_dir: str = "dataset"):
        """Initialize the benchmark runner."""
        self.config_path = config_path
        self.dataset_dir = dataset_dir
        self.benchmark_dir = None
        self.results = {}
        
    def load_config(self) -> List[str]:
        """Load models from config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                return config.get('models', [])
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}")
            sys.exit(1)
    
    def setup_benchmark_directory(self) -> str:
        """Create benchmark directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = Path("results") / f"benchmark_{timestamp}"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir = str(benchmark_dir)
        return self.benchmark_dir
    
    def format_model_name(self, model_id: str) -> str:
        """Convert model ID to human-readable name."""
        # Create a mapping for common model names
        model_mapping = {
            "gemini/gemini-2.5-pro-preview-05-06": "Gemini 2.5 Pro",
            "gemini/gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash",
            "openai/gpt-4.1": "GPT-4.1",
            "openai/gpt-4.1-mini": "GPT-4.1-Mini",
            "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
            "anthropic/claude-3-5-haiku-20241022": "Claude 3.5 Haiku"
        }
        
        return model_mapping.get(model_id, model_id)
    
    def extract_accuracy_from_result(self, result_path: str) -> float:
        """Extract accuracy from evaluation result file."""
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
                return data.get('summary', {}).get('overall_accuracy', 0.0)
        except Exception as e:
            print(f"Error reading result file {result_path}: {e}")
            return 0.0
    
    async def evaluate_model(self, model: str, dataset: TypstBenchDataset) -> Tuple[str, float]:
        """Evaluate a single model and return its accuracy."""
        print(f"\nEvaluating {self.format_model_name(model)}...")
        
        # Create evaluator for this model
        evaluator = TypstBenchEvaluator(
            dataset=dataset,
            model=model,
            output_dir=self.benchmark_dir,
            max_concurrent_requests=3  # Conservative to avoid rate limits
        )
        
        try:
            # Run evaluation
            result_file = await evaluator.run_evaluation()
            
            # Move and rename the result file to follow our naming convention
            original_path = Path(result_file)
            model_filename = model.replace("/", "-").replace(":", "-") + ".json"
            new_path = Path(self.benchmark_dir) / f"{model_filename}"
            
            original_path.rename(new_path)
            
            # Extract accuracy
            accuracy = self.extract_accuracy_from_result(str(new_path))
            
            print(f"✓ {self.format_model_name(model)}: {accuracy:.2%}")
            return model, accuracy
            
        except Exception as e:
            print(f"✗ Error evaluating {self.format_model_name(model)}: {e}")
            return model, 0.0
    
    def print_results_table(self, results: Dict[str, float]):
        """Print results in a formatted table."""
        # Sort results by accuracy (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print()
        print("| {:<30} | {:>10} |".format("Model", "Accuracy"))
        print("|" + "-"*32 + "|" + "-"*12 + "|")
        
        for model, accuracy in sorted_results:
            model_name = self.format_model_name(model)
            print("| {:<30} | {:>9.2%} |".format(model_name, accuracy))
        print()
    
    async def run_benchmark(self, max_samples: int = None, **filter_kwargs):
        """Run the complete benchmark."""
        # Load models from config
        models = self.load_config()
        print(f"Found {len(models)} models in config file")
        
        # Setup benchmark directory
        benchmark_dir = self.setup_benchmark_directory()
        print(f"Results will be saved to: {benchmark_dir}")
        
        # Load dataset
        print(f"Loading dataset from {self.dataset_dir}...")
        dataset = TypstBenchDataset(self.dataset_dir)
        
        # Evaluate each model
        print(f"\nStarting evaluation of {len(models)} models...")
        print("This may take a while depending on the number of samples and models...")
        
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Evaluating {self.format_model_name(model)}...")
            
            try:
                # Create evaluator for this model with custom output directory
                evaluator = TypstBenchEvaluator(
                    dataset=dataset,
                    model=model,
                    output_dir=benchmark_dir,
                    max_concurrent_requests=3
                )
                
                # Prepare filter kwargs for this evaluation
                eval_filter_kwargs = filter_kwargs.copy() if filter_kwargs else {}
                
                # Run evaluation
                start_time = time.time()
                result_file = await evaluator.run_evaluation(
                    filter_kwargs=eval_filter_kwargs,
                    max_samples=max_samples
                )
                
                # Move and rename the result file
                original_path = Path(result_file)
                model_filename = model.replace("/", "-").replace(":", "-") + ".json"
                new_path = Path(benchmark_dir) / model_filename
                
                if original_path != new_path:
                    original_path.rename(new_path)
                
                # Extract accuracy
                accuracy = self.extract_accuracy_from_result(str(new_path))
                self.results[model] = accuracy
                
                elapsed = time.time() - start_time
                print(f"✓ Completed in {elapsed:.1f}s - Accuracy: {accuracy:.2%}")
                
            except Exception as e:
                print(f"✗ Error evaluating {model}: {e}")
                self.results[model] = 0.0
        
        # Print final results table
        self.print_results_table(self.results)
        
        # Save summary results
        summary_file = Path(benchmark_dir) / "benchmark_summary.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": len(models),
            "results": {
                self.format_model_name(model): accuracy 
                for model, accuracy in self.results.items()
            },
            "raw_results": self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nBenchmark complete! Summary saved to {summary_file}")


async def main():
    parser = argparse.ArgumentParser(description="Run TypstBench evaluation on multiple models")
    parser.add_argument("--config", default="all.json", help="Path to config file with model list")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing the dataset")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate per model")
    parser.add_argument("--category", help="Filter samples by category")
    parser.add_argument("--tier", help="Filter samples by tier")
    parser.add_argument("--features", help="Filter samples by features (comma-separated)")
    
    args = parser.parse_args()
    
    # Prepare filter kwargs
    filter_kwargs = {}
    if args.category:
        filter_kwargs["category"] = args.category
    if args.tier:
        filter_kwargs["tier"] = args.tier
    if args.features:
        filter_kwargs["features"] = args.features.split(",")
    
    # Create and run benchmark
    runner = BenchmarkRunner(config_path=args.config, dataset_dir=args.dataset_dir)
    await runner.run_benchmark(max_samples=args.max_samples, **filter_kwargs)


if __name__ == "__main__":
    asyncio.run(main())