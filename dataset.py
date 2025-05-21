import json
import os
import glob
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from typst import TypstRenderer, get_typst_code


class VerifyResult:
    IGNORED = 0
    SUCCESS = 1
    FAILURE = 2

class ComparisonMethod:
    PDF_HASH = "pdf_hash"
    STRING_MATCH = "string_match"
    IMAGE_SIMILARITY = "image_similarity" # New method

class TypstBenchSample:
    """Represents a single sample from the TypstBench dataset."""

    def __init__(self,
                 category: str,
                 metadata: Dict[str, Any],
                 raw_input: str,
                 raw_ground_truth: str,
                 file_path: Optional[str] = None,
                 ignore_verify: bool = False,
                 comparison_method: str = ComparisonMethod.IMAGE_SIMILARITY # Changed type to str to accommodate new method
                 ):
        self.category = category
        self.metadata = metadata
        self.raw_input = raw_input
        self.raw_ground_truth = raw_ground_truth
        self.file_path = file_path
        self.ignore_verify = ignore_verify
        self.comparison_method = comparison_method # Ensure this uses the string value

    def __repr__(self) -> str:
        return f"TypstBenchSample({self.file_path})"
    
    @property
    def task_number(self) -> str:
        """Extract task number from the file path."""
        if self.file_path:
            # Extract the task number from the file name
            filename = os.path.basename(self.file_path)
            # Remove the file extension
            task_number = filename.split('.')[0]
            return task_number
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format."""
        return {
            "category": self.category,
            "metadata": self.metadata,
            "raw_input": self.raw_input,
            "raw_output": self.raw_ground_truth,
            "file_path": self.file_path,
            "comparison_method": self.comparison_method
        }
    
    def get_typst_code(self) -> Optional[str]:
        return get_typst_code(self.raw_ground_truth)

    def verify(self, renderer: Optional[TypstRenderer] = None) -> Tuple[int, Optional[str]]:
        if self.ignore_verify:
            return VerifyResult.IGNORED, None
        
        # For string match or image similarity, we don't need to compile the ground truth during dataset verification
        # as it might not be valid Typst code (e.g. for multiple_choice) or we care about visual output.
        # The actual compilation will happen during evaluation.
        if self.comparison_method == ComparisonMethod.STRING_MATCH:
             return VerifyResult.IGNORED, "Verification skipped for string_match method."

        if self.comparison_method == ComparisonMethod.IMAGE_SIMILARITY and self.category == "multiple_choice":
            return VerifyResult.IGNORED, "Verification skipped for image_similarity on multiple_choice (expects string)."


        typst_code_to_verify = get_typst_code(self.raw_ground_truth)
        if not typst_code_to_verify:
            if self.comparison_method == ComparisonMethod.PDF_HASH or self.comparison_method == ComparisonMethod.IMAGE_SIMILARITY:
                 return VerifyResult.FAILURE, "Ground truth does not contain valid Typst code block for PDF/Image comparison."
            # If it's another method or no specific typst block is expected, we might ignore this.
            # For now, let's assume if it's not string match, it should have a typst block.
            return VerifyResult.IGNORED, "Ground truth does not contain Typst code block, but not strictly required for this method."


        if renderer is None:
            renderer = TypstRenderer()
        
        render_result = renderer.render(typst_code_to_verify)
        
        # Clean up temporary files created by render method if they exist
        if render_result.get("typst_file") and os.path.exists(render_result["typst_file"]):
            os.remove(render_result["typst_file"])
        if render_result.get("pdf_path") and os.path.exists(render_result["pdf_path"]):
            os.remove(render_result["pdf_path"])
            
        if render_result["success"]:    
            return VerifyResult.SUCCESS, None
        else:
            return VerifyResult.FAILURE, render_result["error_output"]
        

class TypstBenchDataset:
    """Loader for the TypstBench dataset."""
    samples: List[TypstBenchSample]

    def __init__(self, base_dir: str = "dataset"):
        """
        Initialize the dataset loader.

        Args:
            base_dir: Base directory containing the dataset files
        """
        self.base_dir = base_dir
        self.samples: list[TypstBenchSample] = []
        self._load_dataset()

    def _parse_sample_file(self, file_path: str) -> Optional[TypstBenchSample]:
        """
        Parse a single sample file.

        Args:
            file_path: Path to the sample file, e.g., "dataset/category/042.md"

        Returns:
            A TypstBenchSample object or None if parsing fails
        """
        category = os.path.basename(os.path.dirname(file_path))
        category_ignore_file = os.path.join(os.path.dirname(file_path), ".ignore-verify")

        ignore_verify = False
        if os.path.exists(category_ignore_file):
            ignore_verify = True

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split the file by triple dashes
            parts = re.split(r'\n---\n', content.strip())

            if len(parts) != 3:
                print(f"Warning: File {file_path} doesn't have the expected 3 sections")
                return None

            # Parse YAML frontmatter
            frontmatter_str = parts[0].strip()
            # Remove leading '---' if present
            if frontmatter_str.startswith('---'):
                frontmatter_str = frontmatter_str[3:].strip()

            metadata = yaml.safe_load(frontmatter_str)
            raw_input = parts[1].strip()
            raw_ground_truth = parts[2].strip()

            ignore_verify = metadata.get("ignore_verify", ignore_verify)

            # Determine evaluation method from metadata, defaulting based on category or to PDF_HASH
            comparison_method_str = metadata.get("comparison_method", None)
            if comparison_method_str:
                if comparison_method_str == "pdf_hash":
                    comparison_method = ComparisonMethod.PDF_HASH
                elif comparison_method_str == "string_match":
                    comparison_method = ComparisonMethod.STRING_MATCH
                elif comparison_method_str == "image_similarity":
                    comparison_method = ComparisonMethod.IMAGE_SIMILARITY
                else:
                    print(f"Warning: Unknown comparison_method '{comparison_method_str}' in {file_path}. Defaulting to IMAGE_SIMILARITY.")
                    comparison_method = ComparisonMethod.IMAGE_SIMILARITY
            elif category == "multiple_choice":
                comparison_method = ComparisonMethod.STRING_MATCH
            else:
                # Default for other categories if not specified
                comparison_method = ComparisonMethod.PDF_HASH


            return TypstBenchSample(
                category=category,
                metadata=metadata,
                raw_input=raw_input,
                raw_ground_truth=raw_ground_truth,
                file_path=file_path,
                ignore_verify=ignore_verify,
                comparison_method=comparison_method
            )

        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None

    def _load_dataset(self) -> None:
        """Load all samples from the dataset directory."""
        sample_files = glob.glob(os.path.join(self.base_dir, "**", "*.md"), recursive=True)

        for file_path in sample_files:
            # Skip files in .archive folders
            if ".archive" in file_path.split(os.sep):
                continue
            sample = self._parse_sample_file(file_path)
            if sample:
                self.samples.append(sample)

    def get_sample_by_id(self, sample_id: str) -> Optional[TypstBenchSample]:
        """
        Get a sample by its ID. (Note: 'id' attribute isn't explicitly defined on TypstBenchSample)
        This method might need adjustment based on how 'id' is intended to be used.
        Assuming it refers to task_number for now.
        """
        for sample in self.samples:
            if sample.task_number == sample_id: # Check against task_number
                return sample
        return None

    def filter_samples(self, **kwargs) -> List[TypstBenchSample]:
        """
        Filter samples by metadata fields.

        Args:
            **kwargs: Field-value pairs to filter on

        Returns:
            List of filtered samples
        """
        filtered = self.samples

        for key, value in kwargs.items():
            if key == 'task_number':
                filtered = [s for s in filtered if s.task_number.startswith(value)] # Use startswith for flexibility
            elif key == 'category':
                filtered = [s for s in filtered if s.category == value]
            elif key == 'tier':
                filtered = [s for s in filtered if s.metadata.get('tier') == value]
            elif key == 'features':
                if isinstance(value, list):
                    # Check if any of the requested features are in the sample's features
                    filtered = [s for s in filtered if any(
                        feat in s.metadata.get('features', []) for feat in value
                    )]
                else:
                    # Check if the single requested feature is in the sample's features
                    filtered = [s for s in filtered if value in s.metadata.get('features', [])]
            elif key == 'comparison_method':
                 filtered = [s for s in filtered if s.comparison_method == value]
            else:
                # Generic filter for other metadata fields
                filtered = [s for s in filtered if s.metadata.get(key) == value]

        return filtered

    def get_all_samples(self) -> List[TypstBenchSample]:
        """Get all samples in the dataset."""
        return self.samples

    def get_stats(self) -> "DatasetStats":
        """
        Calculate statistics for the dataset.

        Returns:
            DatasetStats object with statistics
        """
        total_tasks = len(self.samples)
        categories = {}
        features = {}
        comparison_methods = {}

        for sample in self.samples:
            # Count categories
            if sample.category:
                categories[sample.category] = categories.get(sample.category, 0) + 1

            # Count features
            for feature in sample.metadata.get('features', []):
                features[feature] = features.get(feature, 0) + 1
            
            # Count evaluation methods
            comparison_methods[sample.comparison_method] = comparison_methods.get(sample.comparison_method, 0) + 1


        return DatasetStats(root_dir=self.base_dir,
                            total_tasks=total_tasks, categories=categories, features=features, comparison_methods=comparison_methods)

    def verify(self, renderer: Optional[TypstRenderer] = None) -> List[Tuple[str, int, Optional[str]]]:
        """
        Verify all samples in the dataset.

        Args:
            renderer: Optional TypstRenderer instance for rendering

        Returns:
            List of tuples with sample ID, verification result, and error message (if any)
        """
        results = []
        if renderer is None: # Initialize renderer once if not provided
            renderer = TypstRenderer()

        progress_bar = tqdm(self.samples, desc="Verifying samples", unit="sample")
        for sample in progress_bar:
            result, error = sample.verify(renderer) # Pass the single renderer instance
            results.append((sample.file_path, result, error))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire dataset to a dictionary format."""
        return {
            "info": {
                "name": "TypstBench",
                "version": "0.1.0", # Consider making this dynamic or a class variable
                "sample_count": len(self.samples),
                "base_dir": self.base_dir
            },
            "samples": [sample.to_dict() for sample in self.samples]
        }
    
    def to_jsonl(self, output_path: str, indent: Optional[int] = None) -> None:
        """
        Save the dataset to a JSONL (JSON Lines) file.
        
        Args:
            output_path: Path where the JSONL file will be saved
            indent: Optional indentation for JSON formatting (None for compact format)
            
        Returns:
            None
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in self.samples:
                    # Convert each sample to a dictionary and write as a JSON line
                    json_str = json.dumps(sample.to_dict(), ensure_ascii=False, indent=indent)
                    f.write(json_str + '\n')
            
            print(f"Dataset saved to {output_path} with {len(self.samples)} samples")
        except Exception as e:
            print(f"Error saving dataset to JSONL: {e}")


class DatasetStats:
    """Statistics for the TypstBench dataset."""
    root_dir: str
    total_tasks: int
    categories: Dict[str, int]
    features: Dict[str, int]
    comparison_methods: Dict[str, int]

    def __init__(self, root_dir: str, total_tasks: int = 0, categories: Optional[Dict[str, int]] = None, features: Optional[Dict[str, int]] = None, comparison_methods: Optional[Dict[str, int]] = None):
        self.root_dir = root_dir
        self.total_tasks = total_tasks
        self.categories = categories if categories else {}
        self.features = features if features else {}
        self.comparison_methods = comparison_methods if comparison_methods else {}
    
    def __repr__(self) -> str:
        return (f"DatasetStats(root_dir={self.root_dir}, total_tasks={self.total_tasks}, "
                f"categories={self.categories}, features={self.features}, comparison_methods={self.comparison_methods})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            "root_dir": self.root_dir,
            "total_tasks": self.total_tasks,
            "categories": self.categories,
            "features": self.features,
            "comparison_methods": self.comparison_methods
        }
    
    # terminal output
    def print_stats(self) -> None:
        """Print the dataset statistics."""
        print("Root dir:", self.root_dir)
        print(f"Total tasks: {self.total_tasks}")
        print("Categories:")
        for category, count in self.categories.items():
            print(f"  {category}: {count}")
        print("Features:", self.features)
        print("Evaluation Methods:")
        for method, count in self.comparison_methods.items():
            print(f"  {method}: {count}")
        

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="TypstBench Dataset Management Tool"
    )
    parser.add_argument(
        "command",
        choices=["stats", "verify", "render"],
        help="Command to execute: 'stats' for statistics, 'verify' for dataset verification, 'render' for rendering samples"
    )
    parser.add_argument(
        "task_identifier",
        type=str,
        nargs="?",
        metavar="category/task_number",
        help="Sample ID to render (e.g., 'category/task_number')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all samples in the dataset (only applicable for the 'render' command)"
    )
    parser.add_argument(
        "--dataset-dir", default="dataset", help="Directory containing the dataset"
    )


    args = parser.parse_args()

    dataset = TypstBenchDataset(args.dataset_dir)

    if args.command == "stats":
        stats = dataset.get_stats()
        stats.print_stats()
    elif args.command == "verify":
        results = dataset.verify()

        ignored_count = sum(1 for _, result, _ in results if result == VerifyResult.IGNORED)
        success_count = sum(1 for _, result, _ in results if result == VerifyResult.SUCCESS)
        failure_count = sum(1 for _, result, _ in results if result == VerifyResult.FAILURE)
        print(f"\nVerification Summary:")
        print(f"Total Samples: {len(results)}")
        print(f"Ignored: {ignored_count}")
        print(f"Success: {success_count}")
        print(f"Failure: {failure_count}")
        
        if failure_count > 0:
            print("\nFailures:")
            for file_path, result_code, error_msg in results:
                if result_code == VerifyResult.FAILURE:
                    print(f"  File: {file_path}")
                    print(f"    Error: {error_msg}")
        if failure_count == 0 and success_count > 0 :
             print("All verifiable samples passed verification.")
        elif success_count == 0 and failure_count == 0 and ignored_count > 0:
            print("All samples were ignored for verification.")


    elif args.command == "render":
        renderer = TypstRenderer()
        output_dir = "renders"
        os.makedirs(output_dir, exist_ok=True)

        samples_to_render: List[TypstBenchSample] = []
        if args.all:
            samples_to_render = dataset.get_all_samples()
        elif args.task_identifier:
            try:
                category, task_number = args.task_identifier.split("/")
                samples_to_render = dataset.filter_samples(category=category, task_number=task_number)
                if not samples_to_render:
                    print(f"Error: No samples found for identifier '{args.task_identifier}'")
                    return
            except ValueError:
                print(f"Error: Invalid task_identifier format. Expected 'category/task_number', got '{args.task_identifier}'")
                return
        else:
            print("Error: You must specify either a task identifier (category/task_number) or use the --all flag for the 'render' command.")
            return

        for sample in tqdm(samples_to_render, desc="Rendering samples"):
            if sample.comparison_method == ComparisonMethod.STRING_MATCH:
                print(f"Skipping render for {sample.file_path} (string_match evaluation).")
                continue

            typst_code_to_render = sample.get_typst_code()
            if not typst_code_to_render:
                print(f"Skipping render for {sample.file_path} (no Typst code block in ground truth).")
                continue

            # Construct a unique output path within the 'renders' directory
            # e.g., renders/category/task_number.pdf
            render_output_path = os.path.join(output_dir, sample.category, f"{sample.task_number}.pdf")
            os.makedirs(os.path.dirname(render_output_path), exist_ok=True)

            result = renderer.render(typst_code_to_render, pdf_path=render_output_path)
            if not result["success"]:
                print(f"Error rendering {sample.file_path}: {result.get('error_output', 'Unknown error')}")
                # Clean up failed PDF if it was created
                if os.path.exists(render_output_path):
                    os.remove(render_output_path)
                # Clean up temporary .typ file created by renderer
                if result.get("typst_file") and os.path.exists(result["typst_file"]):
                    os.remove(result["typst_file"])
                continue
            
            # Clean up temporary .typ file created by renderer on success
            if result.get("typst_file") and os.path.exists(result["typst_file"]):
                 os.remove(result["typst_file"])


            # print(f"Rendered {sample.file_path} to {result['pdf_path']}") # result['pdf_path'] is the correct one

if __name__ == "__main__":
    """
    Usage:
        python dataset.py stats [--dataset-dir path/to/dataset]
        python dataset.py verify [--dataset-dir path/to/dataset]
        python dataset.py render <category>/<task_number> [--dataset-dir path/to/dataset]
        python dataset.py render --all [--dataset-dir path/to/dataset]
    """
    main()
