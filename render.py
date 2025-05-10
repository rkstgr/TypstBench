import os
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import shutil
from pathlib import Path

from dataset import TypstBenchDataset, TypstBenchSample
from typst import TypstRenderer, TypstRenderResult


class DatasetRenderer:
    """Renders Typst samples from a dataset to PDF files."""

    def __init__(
        self,
        dataset: TypstBenchDataset,
        output_dir: str = "renders",
        keep_typ_files: bool = False,
        parallel: bool = True,
        max_parallel: int = 4,
    ):
        """
        Initialize the renderer.

        Args:
            dataset: The TypstBench dataset
            output_dir: Base directory for rendered outputs
            keep_typ_files: Whether to keep the .typ files after rendering
            parallel: Whether to render samples in parallel
            max_parallel: Maximum number of parallel rendering processes
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.keep_typ_files = keep_typ_files
        self.parallel = parallel
        self.max_parallel = max_parallel
        self.renderer = TypstRenderer()

        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)

    def _get_output_path(self, sample: TypstBenchSample) -> Tuple[str, str]:
        """
        Determine the output paths for a sample.

        Args:
            sample: The sample to generate paths for

        Returns:
            Tuple of (pdf_path, typ_path)
        """
        # Extract relative path components from the file path
        if sample.file_path:
            # Convert to Path object for easier manipulation
            file_path = Path(sample.file_path)

            # Get the relative path from the dataset base
            rel_path = file_path.relative_to(Path(self.dataset.base_dir))

            # Replace .md extension with .pdf/.typ
            pdf_path = os.path.join(self.output_dir, rel_path.with_suffix(".pdf"))
            typ_path = os.path.join(self.output_dir, rel_path.with_suffix(".typ"))
        else:
            # Fallback if no file path (use sample ID)
            pdf_path = os.path.join(self.output_dir, f"{sample.id}.pdf")
            typ_path = os.path.join(self.output_dir, f"{sample.id}.typ")

        return pdf_path, typ_path

    async def _render_sample(self, sample: TypstBenchSample) -> Dict[str, Any]:
        """
        Render a single sample to PDF.

        Args:
            sample: The sample to render

        Returns:
            Dictionary with render results
        """
        pdf_path, typ_path = self._get_output_path(sample)

        # Create output directory if needed
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        # Save Typst code to file
        with open(typ_path, "w", encoding="utf-8") as f:
            f.write(sample.raw_output)

        # Render with Typst
        result = self.renderer.render_file(typst_file=typ_path, pdf_file=pdf_path)

        # Clean up .typ file if not keeping
        if not self.keep_typ_files and os.path.exists(typ_path):
            os.remove(typ_path)

        # Clean up temporary files
        result.cleanup()

        return {
            "sample_id": sample.id,
            "success": result.success,
            "pdf_path": pdf_path if result.success else None,
            "errors": [str(e) for e in result.errors] if result.errors else [],
            "file_path": sample.file_path,
        }

    async def render_samples(
        self, samples: List[TypstBenchSample]
    ) -> List[Dict[str, Any]]:
        """
        Render multiple samples, potentially in parallel.

        Args:
            samples: List of samples to render

        Returns:
            List of render results dictionaries
        """
        if self.parallel:
            # Create task groups with limited concurrency
            sem = asyncio.Semaphore(self.max_parallel)

            async def _render_with_semaphore(sample):
                async with sem:
                    return await self._render_sample(sample)

            tasks = [_render_with_semaphore(sample) for sample in samples]
            return await asyncio.gather(*tasks)
        else:
            # Render samples sequentially
            results = []
            for sample in samples:
                result = await self._render_sample(sample)
                results.append(result)
            return results

    async def render_dataset(
        self, filter_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Render samples from the dataset with optional filtering.

        Args:
            filter_kwargs: Arguments to filter samples (passed to dataset.filter_samples)

        Returns:
            Dictionary with rendering summary and results
        """
        # Filter samples if needed
        if filter_kwargs:
            samples = self.dataset.filter_samples(**filter_kwargs)
        else:
            samples = self.dataset.get_all_samples()

        print(f"Rendering {len(samples)} samples to {self.output_dir}")

        # Render all samples
        results = await self.render_samples(samples)

        # Calculate summary metrics
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        success_rate = successful / total if total > 0 else 0

        # Calculate tier-specific metrics if available
        tier_metrics = {}
        tier_samples = {}

        for sample in samples:
            tier = sample.metadata.get("tier")
            if tier:
                if tier not in tier_samples:
                    tier_samples[tier] = []
                tier_samples[tier].append(sample.id)

        for tier, sample_ids in tier_samples.items():
            tier_results = [r for r in results if r["sample_id"] in sample_ids]
            tier_total = len(tier_results)
            tier_successful = sum(1 for r in tier_results if r["success"])
            tier_success_rate = tier_successful / tier_total if tier_total > 0 else 0

            tier_metrics[tier] = {
                "total": tier_total,
                "successful": tier_successful,
                "success_rate": tier_success_rate,
            }

        # Compile summary metrics
        summary = {
            "total_samples": total,
            "successful_samples": successful,
            "overall_success_rate": success_rate,
            "tier_metrics": tier_metrics,
            "output_dir": self.output_dir,
        }

        return {"summary": summary, "results": results}


async def main():
    parser = argparse.ArgumentParser(
        description="Render TypstBench dataset samples to PDF"
    )
    parser.add_argument(
        "--dataset-dir", default="dataset", help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output-dir", default="renders", help="Directory to save rendered PDFs"
    )
    parser.add_argument("--tier", help="Filter samples by tier (e.g., 'basic')")
    parser.add_argument(
        "--features", help="Filter samples by features (comma-separated)"
    )
    parser.add_argument(
        "--difficulty", type=int, help="Filter samples by difficulty level"
    )
    parser.add_argument("--task-type", help="Filter samples by task type")
    parser.add_argument(
        "--keep-typ", action="store_true", help="Keep the .typ files after rendering"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Render samples sequentially instead of in parallel",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum concurrent rendering processes",
    )

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

    # Initialize renderer
    renderer = DatasetRenderer(
        dataset=dataset,
        output_dir=args.output_dir,
        keep_typ_files=args.keep_typ,
        parallel=not args.sequential,
        max_parallel=args.max_parallel,
    )

    # Render dataset
    results = await renderer.render_dataset(
        filter_kwargs=filter_kwargs if filter_kwargs else None
    )

    # Print summary
    summary = results["summary"]
    print("\nRendering Summary:")
    print(f"Total Samples: {summary['total_samples']}")
    print(
        f"Success Rate: {summary['overall_success_rate']:.2%} ({summary['successful_samples']}/{summary['total_samples']})"
    )

    if summary["tier_metrics"]:
        print("\nSuccess Rate by Tier:")
        for tier, metrics in summary["tier_metrics"].items():
            print(
                f"  {tier}: {metrics['success_rate']:.2%} ({metrics['successful']}/{metrics['total']})"
            )

    # List failed samples
    failed_results = [r for r in results["results"] if not r["success"]]
    if failed_results:
        print(f"\nFailed Samples ({len(failed_results)}):")
        for result in failed_results:
            print(
                f"  {result['sample_id']} - Errors: {', '.join(result['errors'][:1])}..."
            )


if __name__ == "__main__":
    asyncio.run(main())
