import os
import json
from pathlib import Path
import time
import argparse
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, TypedDict
import litellm
from litellm.types.utils import ModelResponse, Choices
from dataset import ComparisonMethod, TypstBenchDataset, TypstBenchSample # ComparisonMethod imported
from typst import TypstRenderer, get_typst_code
import subprocess # Added
import shutil # Added
import tempfile # Added

# --- Constants for Image Similarity ---
IMAGE_SIMILARITY_METRIC = "AE"  # Absolute Error (count of different pixels)
IMAGE_SIMILARITY_FUZZ = "1%"    # Fuzz factor for ImageMagick (e.g., "1%")
IMAGE_SIMILARITY_THRESHOLD = 0  # Max allowed different pixels to be considered a match

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

def _run_imagemagick_compare(
    target_pdf_path: str,
    model_pdf_path: str,
    diff_output_path: Optional[str],
    metric: str,
    fuzz: str
) -> Tuple[Optional[int], Optional[str], int]:
    """
    Runs ImageMagick's compare command on two PDF files (first page).

    Args:
        target_pdf_path: Path to the target (ground truth) PDF.
        model_pdf_path: Path to the model's output PDF.
        diff_output_path: Optional path to save the diff image. If None, no diff image is saved.
        metric: The metric to use for comparison (e.g., "AE").
        fuzz: The fuzz factor for comparison (e.g., "1%").

    Returns:
        A tuple containing:
        - pixel_difference (Optional[int]): The number of differing pixels, or None on error.
        - error_message (Optional[str]): Error message from stderr if any.
        - return_code (int): The return code of the magick compare command.
    """
    # Ensure ImageMagick is available (though typically checked before calling this)
    if not shutil.which("magick"):
        return None, "ImageMagick 'magick' command not found.", -1 # Custom return code for not found

    cmd = [
        "magick", "compare",
        "-metric", metric,
        "-fuzz", fuzz,
        f"{target_pdf_path}[0]",  # Compare first page
        f"{model_pdf_path}[0]",   # Compare first page
    ]
    # If diff_output_path is provided, ImageMagick saves the diff there.
    # If not, ImageMagick needs a placeholder for the diff output argument.
    # 'null:' is a common cross-platform way to discard output for ImageMagick.
    cmd.append(diff_output_path if diff_output_path else "null:")

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        # ImageMagick `compare -metric AE` outputs the difference to stderr.
        # Return code 0 means images are similar (within fuzz).
        # Return code 1 means images are different.
        # Return code >1 means an error occurred.
        
        stderr_output = process.stderr.strip()
        pixel_difference: Optional[int] = None

        if process.returncode <= 1: # 0 or 1 indicates successful comparison execution
            try:
                # The actual numeric difference is usually the only thing on stderr
                # or the first number if there are other warnings.
                # Sometimes it might be a float (e.g. for PSNR), so parse as float then int.
                pixel_difference = int(float(stderr_output.split()[0]))
            except (ValueError, IndexError):
                # If parsing fails, it might be an error message or unexpected output
                if process.returncode == 0 and not stderr_output: # Identical, AE is 0, stderr might be empty
                    pixel_difference = 0
                elif stderr_output: # If there's output, treat as error for parsing
                     return None, f"Failed to parse pixel difference from ImageMagick output: {stderr_output}", process.returncode

        return pixel_difference, stderr_output if process.returncode > 1 else None, process.returncode

    except FileNotFoundError: # Should be caught by shutil.which earlier, but as a fallback
        return None, "ImageMagick 'magick' command not found during execution.", -1
    except Exception as e:
        return None, f"Exception during ImageMagick compare: {str(e)}", -2 # Custom return code for other exceptions


def compare_outputs(model_output: str,
                    target_output: str,
                    comparison_method: str,
                    debug_artifacts_dir: Optional[Path] = None
                    ) -> ComparisonResult:
    if comparison_method == ComparisonMethod.STRING_MATCH:
        is_correct = model_output.strip() == target_output.strip()
        return ComparisonResult(
            is_correct=is_correct,
            error_message=None,
            comparison_method=ComparisonMethod.STRING_MATCH
        )
    
    elif comparison_method == ComparisonMethod.PDF_HASH:
        # --- Model PDF Rendering for Hash ---
        model_typst_code = get_typst_code(model_output)
        if not model_typst_code:
            return ComparisonResult(
                is_correct=False,
                error_message="Model output is not valid Typst code",
                comparison_method="compile_check_model_hash" # More specific
            )
        
        # For PDF hash, we always use temporary files for rendering, then hash.
        # Debug artifacts dir is only for saving the model's PDF if hashes don't match.
        renderer = TypstRenderer()
        model_render_result = renderer.render(model_typst_code) # Renders to temp PDF

        if not model_render_result["success"]:
            if model_render_result.get("typst_file") and os.path.exists(model_render_result["typst_file"]): # Cleanup temp .typ
                os.remove(model_render_result["typst_file"])
            if model_render_result.get("pdf_path") and os.path.exists(model_render_result["pdf_path"]): # Cleanup temp .pdf
                os.remove(model_render_result["pdf_path"])
            return ComparisonResult(
                is_correct=False,
                error_message=f"Model output failed to render for PDF hash: {model_render_result.get('error_output', 'Unknown error')}",
                comparison_method=ComparisonMethod.PDF_HASH
            )
        
        actual_model_pdf_path = Path(model_render_result["pdf_path"])
        actual_model_typ_path = Path(model_render_result["typst_file"])

        # --- Target PDF Rendering for Hash ---
        target_typst_code = get_typst_code(target_output)
        if not target_typst_code:
            if os.path.exists(actual_model_pdf_path): os.remove(actual_model_pdf_path) # Cleanup model's temp PDF
            if os.path.exists(actual_model_typ_path): os.remove(actual_model_typ_path) # Cleanup model's temp .typ
            return ComparisonResult(
                is_correct=False,
                error_message="Target output is not valid Typst code for PDF hash",
                comparison_method="dataset_error_target_hash" # More specific
            )
        
        target_render_result = renderer.render(target_typst_code) # Renders to temp PDF
        if not target_render_result["success"]:
            if os.path.exists(actual_model_pdf_path): os.remove(actual_model_pdf_path)
            if os.path.exists(actual_model_typ_path): os.remove(actual_model_typ_path)
            if target_render_result.get("typst_file") and os.path.exists(target_render_result["typst_file"]): # Cleanup temp .typ
                os.remove(target_render_result["typst_file"])
            if target_render_result.get("pdf_path") and os.path.exists(target_render_result["pdf_path"]): # Cleanup temp .pdf
                os.remove(target_render_result["pdf_path"])
            return ComparisonResult(
                is_correct=False,
                error_message=f"Target output failed to render for PDF hash: {target_render_result.get('error_output', 'Unknown error')}",
                comparison_method=ComparisonMethod.PDF_HASH
            )
        
        actual_target_pdf_path = Path(target_render_result["pdf_path"])
        actual_target_typ_path = Path(target_render_result["typst_file"])

        # --- Calculate Hashes ---
        model_hash = calculate_file_md5(str(actual_model_pdf_path))
        target_hash = calculate_file_md5(str(actual_target_pdf_path))

        # --- Cleanup Temporary Rendered PDFs and TYP files for Hashing ---
        if os.path.exists(actual_model_pdf_path): os.remove(actual_model_pdf_path)
        if os.path.exists(actual_model_typ_path): os.remove(actual_model_typ_path)
        if os.path.exists(actual_target_pdf_path): os.remove(actual_target_pdf_path)
        if os.path.exists(actual_target_typ_path): os.remove(actual_target_typ_path)
        
        if not model_hash or not target_hash:
            return ComparisonResult(
                is_correct=False,
                error_message="Failed to calculate PDF hash (PDFs were temporary and are now deleted).",
                comparison_method=ComparisonMethod.PDF_HASH
            )
        
        is_correct = model_hash == target_hash
        error_msg_hash = None
        if not is_correct and debug_artifacts_dir:
            # Re-render model PDF to debug_artifacts_dir if hashes don't match
            os.makedirs(debug_artifacts_dir, exist_ok=True)
            debug_model_pdf_path = debug_artifacts_dir / "model_output_hash_fail.pdf"
            # We need the original model_typst_code to re-render
            renderer.render(model_typst_code, pdf_path=str(debug_model_pdf_path)) # typ_path can be None
            error_msg_hash = f"PDF hashes do not match. Model output saved to: {str(debug_model_pdf_path)}"
        elif not is_correct:
            error_msg_hash = "PDF hashes do not match."
            
        return ComparisonResult(
            is_correct=is_correct,
            error_message=error_msg_hash,
            comparison_method=ComparisonMethod.PDF_HASH
        )

    elif comparison_method == ComparisonMethod.IMAGE_SIMILARITY:
        # 1. Check if ImageMagick is available
        if not shutil.which("magick"):
            return ComparisonResult(
                is_correct=False,
                error_message="ImageMagick 'magick' command not found. Skipping image comparison.",
                comparison_method=ComparisonMethod.IMAGE_SIMILARITY
            )

        # 2. Get Typst code for model and target
        model_typst_code = get_typst_code(model_output)
        if not model_typst_code:
            return ComparisonResult(is_correct=False, error_message="Model output is not valid Typst code for image comparison.", comparison_method="compile_check_model_img")

        target_typst_code = get_typst_code(target_output)
        if not target_typst_code:
            return ComparisonResult(is_correct=False, error_message="Target output (ground truth) is not valid Typst code for image comparison.", comparison_method="dataset_error_target_img")

        renderer = TypstRenderer()
        
        # Paths for PDFs and TYPs from renderer
        # These will be populated by the render calls
        rendered_model_pdf_path: Optional[Path] = None
        rendered_model_typ_path: Optional[Path] = None
        rendered_target_pdf_path: Optional[Path] = None
        rendered_target_typ_path: Optional[Path] = None

        # Determine if output files should be temporary or in debug_artifacts_dir
        # _*_pdf_render_path are the paths to pass to renderer.render()
        # If None, renderer makes them temporary.
        _model_pdf_render_path: Optional[str] = None
        _target_pdf_render_path: Optional[str] = None
        
        # Also save .typ files to debug_artifacts_dir if it's provided
        _model_typ_debug_path: Optional[Path] = None
        _target_typ_debug_path: Optional[Path] = None


        if debug_artifacts_dir:
            os.makedirs(debug_artifacts_dir, exist_ok=True)
            _model_pdf_render_path = str(debug_artifacts_dir / "model_output.pdf")
            _target_pdf_render_path = str(debug_artifacts_dir / "target_output.pdf")
            _model_typ_debug_path = debug_artifacts_dir / "model_output.typ"
            _target_typ_debug_path = debug_artifacts_dir / "target_output.typ"
            
            with open(_model_typ_debug_path, "w", encoding="utf-8") as f:
                f.write(model_typst_code)
            with open(_target_typ_debug_path, "w", encoding="utf-8") as f:
                f.write(target_typst_code)

        try:
            # --- Render Model PDF ---
            # Pass model_typst_code. If _model_typ_debug_path is set, renderer can use it if modified to do so,
            # or we use it for our records. Current renderer.render takes typst_code.
            # We pass pdf_path for explicit save location if debug_artifacts_dir is used.
            model_render_result = renderer.render(typst_code=model_typst_code, pdf_path=_model_pdf_render_path)
            if not model_render_result["success"]:
                return ComparisonResult(
                    is_correct=False,
                    error_message=f"Model output failed to render for image comparison: {model_render_result.get('error_output', 'Unknown error')}",
                    comparison_method=ComparisonMethod.IMAGE_SIMILARITY
                )
            rendered_model_pdf_path = Path(model_render_result["pdf_path"])
            rendered_model_typ_path = Path(model_render_result["typst_file"]) # .typ used by renderer

            # --- Render Target PDF ---
            target_render_result = renderer.render(typst_code=target_typst_code, pdf_path=_target_pdf_render_path)
            if not target_render_result["success"]:
                return ComparisonResult(
                    is_correct=False,
                    error_message=f"Target output failed to render for image comparison: {target_render_result.get('error_output', 'Unknown error')}",
                    comparison_method=ComparisonMethod.IMAGE_SIMILARITY
                )
            rendered_target_pdf_path = Path(target_render_result["pdf_path"])
            rendered_target_typ_path = Path(target_render_result["typst_file"]) # .typ used by renderer

            # --- Perform ImageMagick Comparison ---
            diff_image_file_path_str: Optional[str] = None
            if debug_artifacts_dir:
                # This path is where magick compare will save the diff image
                diff_image_file_path_str = str(debug_artifacts_dir / "diff_output.png")

            pixel_diff, magick_error_msg, magick_return_code = _run_imagemagick_compare(
                str(rendered_target_pdf_path),
                str(rendered_model_pdf_path),
                diff_image_file_path_str,
                IMAGE_SIMILARITY_METRIC,
                IMAGE_SIMILARITY_FUZZ
            )

            # --- Process Comparison Result ---
            final_error_message: Optional[str] = None
            is_correct = False

            if magick_return_code < 0: # Our custom codes for pre-check failures or exceptions
                 final_error_message = f"ImageMagick pre-check failed or exception: {magick_error_msg or 'Failed before comparison'}"
            elif magick_return_code > 1 or pixel_diff is None : # ImageMagick command error during compare
                final_error_message = f"ImageMagick comparison process error: {magick_error_msg or 'Unknown ImageMagick error'}"
                # If diff image was meant to be created but failed, it might be corrupted or empty
                if diff_image_file_path_str and os.path.exists(diff_image_file_path_str):
                    is_empty_diff = os.path.getsize(diff_image_file_path_str) == 0
                    if is_empty_diff : # remove empty diff file on error
                         os.remove(diff_image_file_path_str)
            else: # ImageMagick compare ran (exit code 0 or 1)
                is_correct = pixel_diff <= IMAGE_SIMILARITY_THRESHOLD
                if not is_correct:
                    final_error_message = f"Images differ by {pixel_diff} pixels (threshold: {IMAGE_SIMILARITY_THRESHOLD})."
                    if diff_image_file_path_str and os.path.exists(diff_image_file_path_str):
                        final_error_message += f" Diff image: {diff_image_file_path_str}"
                    # If no debug_artifacts_dir, diff_image_file_path_str was None, so no diff saved.
                else: # is_correct is True
                    # If a diff image was created in debug_artifacts_dir but images are correct, remove it.
                    if diff_image_file_path_str and os.path.exists(diff_image_file_path_str):
                        os.remove(diff_image_file_path_str)
            
            return ComparisonResult(
                is_correct=is_correct,
                error_message=final_error_message,
                comparison_method=ComparisonMethod.IMAGE_SIMILARITY
            )

        finally:
            # --- Cleanup ---
            # If _model_pdf_render_path was None, it means renderer used a temporary PDF.
            # rendered_model_pdf_path holds the path to that (potentially temporary) PDF.
            if rendered_model_pdf_path and not _model_pdf_render_path and os.path.exists(rendered_model_pdf_path):
                os.remove(rendered_model_pdf_path)
            # The .typ file created by renderer is always temporary unless we modify renderer to take explicit .typ path for its internal use.
            # For now, assume renderer's .typ (Path(model_render_result["typst_file"])) is always cleaned up if it was temporary.
            # If _model_typ_debug_path is set, that's our copy in debug_artifacts_dir.
            # The renderer's internal .typ file (rendered_model_typ_path) should be cleaned if it's not the same as _model_typ_debug_path
            if rendered_model_typ_path and os.path.exists(rendered_model_typ_path):
                if not (_model_typ_debug_path and Path(rendered_model_typ_path) == _model_typ_debug_path):
                     os.remove(rendered_model_typ_path)


            if rendered_target_pdf_path and not _target_pdf_render_path and os.path.exists(rendered_target_pdf_path):
                os.remove(rendered_target_pdf_path)
            if rendered_target_typ_path and os.path.exists(rendered_target_typ_path):
                if not (_target_typ_debug_path and Path(rendered_target_typ_path) == _target_typ_debug_path):
                    os.remove(rendered_target_typ_path)
    
    else:
        raise ValueError(f"Unknown comparison method: {comparison_method}")


class TypstBenchEvaluator:
    """Evaluates LLM performance on the TypstBench dataset."""

    def __init__(
        self,
        dataset: TypstBenchDataset,
        model: str,
        api_key: Optional[str] = None,
        artifacts_dir: str = "pdf_artifacts", # This is the root for debug artifacts
        prompt_template: str = "Task type: {task_type}\nTask:\n{input}",
        max_concurrent_requests: int = 5
    ):
        self.dataset = dataset
        self.model = model
        self.prompt_template = prompt_template
        # artifacts_dir is the base for this evaluation run's artifacts
        # debug_artifacts_dir for compare_outputs will be a subfolder per sample
        self.artifacts_dir = Path(artifacts_dir) 
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

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

        # Create the root artifacts directory for this run
        os.makedirs(self.artifacts_dir, exist_ok=True)


    def _format_prompt(self, sample: TypstBenchSample) -> str:
        """Format the prompt for a sample using the prompt template."""
        return self.prompt_template.format(task_type=sample.category, input=sample.raw_input)

    async def _evaluate_sample(self, sample: TypstBenchSample) -> EvaluateTaskResult:
        """
        Evaluate a single sample using the LLM.
        """
        async with self.semaphore:
            prompt = self._format_prompt(sample)
            start_time = time.time()

            # Define the specific debug directory for this sample
            sample_debug_artifacts_dir = self.artifacts_dir / sample.category / sample.task_number
            # No need to create it here; compare_outputs will do it if needed.

            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        { "role": "system", "content": self.SYSTEM_PROMPT},
                        { "role": "user", "content": prompt}
                        ],
                    temperature=self.TEMPERATURE,
                    max_tokens=1024,
                )
                assert isinstance(response, ModelResponse)
                assert isinstance(response.choices[0], Choices)
                content = response.choices[0].message.content
                assert content
                model_output_text = content.strip()

                # Pass sample_debug_artifacts_dir to compare_outputs
                comparison_result_data = compare_outputs(
                    model_output_text,
                    sample.raw_ground_truth,
                    sample.comparison_method, # This is a string from TypstBenchSample
                    debug_artifacts_dir=sample_debug_artifacts_dir # Pass the specific path
                )

                return EvaluateTaskResult(
                    task=sample.file_path or "unknown_task",
                    prompt=prompt,
                    expected_output=sample.raw_ground_truth,
                    model_output=model_output_text,
                    is_correct=comparison_result_data["is_correct"],
                    latency_seconds=time.time() - start_time,
                    model=self.model,
                    metadata=sample.metadata,
                    comparison_result=comparison_result_data,
                )

            except Exception as e:
                # Ensure comparison_result structure is consistent on error
                error_comparison_result = ComparisonResult(
                    is_correct=False, 
                    error_message=str(e), 
                    comparison_method="evaluation_error"
                )
                return EvaluateTaskResult(
                    task=sample.file_path or "unknown_task",
                    prompt=prompt,
                    expected_output=sample.raw_ground_truth,
                    model_output="", # No model output if exception during call
                    is_correct=False,
                    latency_seconds=time.time() - start_time,
                    model=self.model,
                    metadata=sample.metadata,
                    comparison_result=error_comparison_result
                )

    async def evaluate_samples(self, samples: List[TypstBenchSample]) -> List[EvaluateTaskResult]: # Return type updated
        """
        Evaluate multiple samples in parallel.
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

        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]

        print(f"Evaluating {len(samples)} samples with model: {self.model}")
        results = await self.evaluate_samples(samples)

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        accuracy = correct / total if total > 0 else 0
        avg_latency = sum(r["latency_seconds"] for r in results if "latency_seconds" in r) / total if total > 0 else 0


        tier_metrics: Dict[str, Dict[str, Any]] = {}
        tiers = sorted(list(set(r["metadata"].get("tier") for r in results if "tier" in r["metadata"] and r["metadata"].get("tier") is not None)))


        for tier_value in tiers:
            tier_results = [r for r in results if r["metadata"].get("tier") == tier_value]
            tier_total = len(tier_results)
            tier_correct = sum(1 for r in tier_results if r["is_correct"])
            tier_accuracy = tier_correct / tier_total if tier_total > 0 else 0
            tier_metrics[tier_value] = {
                "total": tier_total,
                "correct": tier_correct,
                "accuracy": tier_accuracy
            }

        feature_metrics: Dict[str, Dict[str, Any]] = {}
        all_features: set[str] = set()
        for r in results:
            features_list = r["metadata"].get("features", [])
            if isinstance(features_list, list): # Ensure it's a list
                all_features.update(f for f in features_list if f is not None)
        
        sorted_features = sorted(list(all_features))

        for feature_value in sorted_features:
            feature_results = [r for r in results if feature_value in r["metadata"].get("features", [])]
            feature_total = len(feature_results)
            feature_correct = sum(1 for r in feature_results if r["is_correct"])
            feature_accuracy = feature_correct / feature_total if feature_total > 0 else 0
            feature_metrics[feature_value] = {
                "total": feature_total,
                "correct": feature_correct,
                "accuracy": feature_accuracy
            }
        
        summary_data = EvaluationRunSummary(
            model=self.model,
            total_samples=total,
            correct_samples=correct,
            overall_accuracy=accuracy,
            average_latency_seconds=avg_latency,
            tier_metrics=tier_metrics,
            feature_metrics=feature_metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_prompt=self.SYSTEM_PROMPT,
            temperature=self.TEMPERATURE,
        )
        return EvaluationRunResults(summary=summary_data, results_per_task=results)


    async def run_evaluation(
        self,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None
    ) -> EvaluationRunResults:
        return await self.evaluate_dataset(filter_kwargs, max_samples)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on the TypstBench dataset")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model identifier for litellm")
    parser.add_argument("--api-key", help="API key for the model provider (optional, can be set via ENV VARS)")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing the dataset")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--category", help="Filter samples by category (e.g., 'multiple_choice')")
    parser.add_argument("--tier", help="Filter samples by tier (e.g., 'basic')")
    parser.add_argument("--features", help="Filter samples by features (comma-separated)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests")
    parser.add_argument("--task", help="Path to a specific task to evaluate (e.g. basic_elements/001.md or category/task_number like multiple_choice/001)")
    

    args = parser.parse_args()

    dataset = TypstBenchDataset(args.dataset_dir)
    filter_kwargs = {}
    if args.category: filter_kwargs["category"] = args.category
    if args.tier: filter_kwargs["tier"] = args.tier
    if args.features: filter_kwargs["features"] = args.features.split(",")
    
    if args.task:
        # Allow task to be specified as "category/task_number" or "category/task_file.md"
        task_parts = args.task.split("/")
        if len(task_parts) == 2:
            filter_kwargs["category"] = task_parts[0]
            filter_kwargs["task_number"] = task_parts[1].split('.')[0] # Get number before .md
        else:
            print(f"Warning: Invalid task format '{args.task}'. Expected 'category/task_number' or 'category/task_file.md'.")


    # --- Output Directory Setup ---
    # Ensure model name is filesystem-friendly
    safe_model_name = args.model.replace("/", "_").replace(":", "-").replace(".", "_")
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    # Results will be stored in: output_dir / safe_model_name / timestamp_str
    # Artifacts for a run will be in: output_dir / safe_model_name / timestamp_str / artifacts / category / task_num / ...
    
    run_results_base_dir = Path(args.output_dir) / safe_model_name / timestamp_str
    run_artifacts_dir = run_results_base_dir / "artifacts" # Specific subdir for PDF/PNG artifacts
    
    os.makedirs(run_results_base_dir, exist_ok=True)
    # run_artifacts_dir will be created by TypstBenchEvaluator or by compare_outputs if needed

    evaluator = TypstBenchEvaluator(
        dataset=dataset,
        model=args.model,
        api_key=args.api_key,
        max_concurrent_requests=args.concurrency,
        artifacts_dir=str(run_artifacts_dir), # Pass the dedicated artifacts dir for this run
    )

    eval_results_data = await evaluator.run_evaluation(
        filter_kwargs=filter_kwargs if filter_kwargs else None,
        max_samples=args.max_samples
    )

    # Save results JSON in the run_results_base_dir
    results_json_file = run_results_base_dir / "results.json"
    with open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(eval_results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults JSON saved to: {results_json_file}")
    print(f"Debug artifacts (if any) for this run are in subdirectories under: {run_artifacts_dir}")

    summary = eval_results_data["summary"]
    print("\nEvaluation Summary:")
    print(f"Model: {summary['model']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%} ({summary['correct_samples']}/{summary['total_samples']})")
    print(f"Average Latency: {summary['average_latency_seconds']:.2f} seconds")

    if summary['tier_metrics']:
        print("\nAccuracy by Tier:")
        for tier_name, metrics in summary['tier_metrics'].items():
            print(f"  {tier_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    if summary['feature_metrics']:
        print("\nAccuracy by Feature:")
        for feature_name, metrics in summary['feature_metrics'].items():
            print(f"  {feature_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    asyncio.run(main())
