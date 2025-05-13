from dataclasses import dataclass
import json
import os
import glob
import yaml
import re
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from typst import TypstRenderer, get_typst_code

@dataclass
class VerifyResult:
    IGNORED = 0
    SUCCESS = 1
    FAILURE = 2

class TypstBenchSample:
    """Represents a single sample from the TypstBench dataset."""

    def __init__(self,
                 category: str,
                 metadata: Dict[str, Any],
                 raw_input: str,
                 raw_ground_truth: str,
                 file_path: Optional[str] = None,
                 ignore_verify: bool = False
                 ):
        self.category = category
        self.metadata = metadata
        self.raw_input = raw_input
        self.raw_ground_truth = raw_ground_truth
        self.file_path = file_path
        self.ignore_verify = ignore_verify

    def __repr__(self) -> str:
        return f"TypstBenchSample({self.file_path})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format."""
        return {
            "category": self.category,
            "metadata": self.metadata,
            "raw_input": self.raw_input,
            "raw_output": self.raw_ground_truth,
            "file_path": self.file_path
        }
    
    def get_typst_code(self) -> Optional[str]:
        return get_typst_code(self.raw_ground_truth)
    
    def get_path(self, root=None, suffix=None):
        """
        Manipulate the file path by optionally changing the root directory and/or file suffix.
        
        Args:
            root: New root directory to replace the current one (if provided)
            suffix: New file suffix to replace the current one (if provided)
            
        Returns:
            Modified file path
        """
        # Split path into parts
        path_parts = self.file_path.split(os.sep)
        
        # Replace root if specified
        if root:
            path_parts[0] = root
        
        # Replace suffix if specified
        if suffix and len(path_parts) > 0:
            filename = path_parts[-1]
            # Find the last dot to replace the extension
            last_dot = filename.rfind('.')
            if last_dot != -1:
                # Replace existing suffix
                path_parts[-1] = filename[:last_dot] + suffix
            else:
                # No existing suffix, just append
                path_parts[-1] = filename + suffix
        
        # Join path parts back into a path
        return os.path.join(*path_parts)

    def verify(self, renderer: Optional[TypstRenderer] = None) -> Tuple[int, Optional[str]]:
        if self.ignore_verify:
            return VerifyResult.IGNORED, None
        
        if renderer is None:
            renderer = TypstRenderer()
        
        render_result = renderer.render(get_typst_code(self.raw_ground_truth))
        os.remove(render_result.typst_file)
        os.remove(render_result.pdf_path)
        if render_result.success:    
            return VerifyResult.SUCCESS, None
        else:
            return VerifyResult.FAILURE, render_result.error_output
        

        

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

            return TypstBenchSample(
                category=category,
                metadata=metadata,
                raw_input=raw_input,
                raw_ground_truth=raw_ground_truth,
                file_path=file_path,
                ignore_verify=ignore_verify
            )

        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None

    def _load_dataset(self) -> None:
        """Load all samples from the dataset directory."""
        sample_files = glob.glob(os.path.join(self.base_dir, "**", "*.md"), recursive=True)

        for file_path in sample_files:
            sample = self._parse_sample_file(file_path)
            if sample:
                self.samples.append(sample)

    def get_sample_by_id(self, sample_id: str) -> Optional[TypstBenchSample]:
        """
        Get a sample by its ID.

        Args:
            sample_id: The ID of the sample to retrieve

        Returns:
            The sample or None if not found
        """
        for sample in self.samples:
            if sample.id == sample_id:
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
            if key == 'id':
                filtered = [s for s in filtered if s.id == value]
            elif key == 'tier':
                filtered = [s for s in filtered if s.metadata.get('tier') == value]
            elif key == 'difficulty':
                filtered = [s for s in filtered if s.metadata.get('difficulty') == value]
            elif key == 'task_type':
                filtered = [s for s in filtered if s.metadata.get('task_type') == value]
            elif key == 'features':
                if isinstance(value, list):
                    # Check if any of the requested features are in the sample's features
                    filtered = [s for s in filtered if any(
                        feat in s.metadata.get('features', []) for feat in value
                    )]
                else:
                    # Check if the single requested feature is in the sample's features
                    filtered = [s for s in filtered if value in s.metadata.get('features', [])]
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

        for sample in self.samples:
            # Count categories
            if sample.category:
                categories[sample.category] = categories.get(sample.category, 0) + 1

            # Count features
            for feature in sample.metadata.get('features', []):
                features[feature] = features.get(feature, 0) + 1

        return DatasetStats(root_dir=self.base_dir,
                            total_tasks=total_tasks, categories=categories, features=features)

    def verify(self, renderer: Optional[TypstRenderer] = None) -> List[Tuple[str, int, Optional[str]]]:
        """
        Verify all samples in the dataset.

        Args:
            renderer: Optional TypstRenderer instance for rendering

        Returns:
            List of tuples with sample ID, verification result, and error message (if any)
        """
        results = []
        progress_bar = tqdm(self.samples, desc="Verifying samples", unit="sample")
        for sample in progress_bar:
            result, error = sample.verify(renderer)
            results.append((sample.file_path, result, error))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire dataset to a dictionary format."""
        return {
            "info": {
                "name": "TypstBench",
                "version": "0.1.0",
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

    def __init__(self, root_dir: str, total_tasks: int = 0, categories: Dict[str, int] = None, features: Dict[str, int] = None): # ty: ignore[invalid-parameter-default]
        self.root_dir = root_dir
        self.total_tasks = total_tasks
        self.categories = categories if categories else {}
        self.features = features if features else {}
    
    def __repr__(self) -> str:
        return f"DatasetStats(root_dir={self.root_dir}, total_tasks={self.total_tasks}, categories={self.categories}, features={self.features})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary format."""
        return {
            "root_dir": self.root_dir,
            "total_tasks": self.total_tasks,
            "categories": self.categories,
            "features": self.features
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
        

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="TypstBench Dataset Management Tool"
    )
    parser.add_argument(
        "command",
        choices=["stats", "verify"],
        help="Command to execute: 'stats' for statistics, 'verify' for dataset verification"
    )

    args = parser.parse_args()

    if args.command == "stats":
        # Load the dataset and print statistics
        dataset = TypstBenchDataset()
        stats = dataset.get_stats()
        stats.print_stats()
    elif args.command == "verify":
        dataset = TypstBenchDataset()
        results = dataset.verify()

        """ Print verification results 
        Ignored: number
        Success: number
        Failure: number

        Failures:
          ...
        """
        ignored_count = sum(1 for _, result, _ in results if result == VerifyResult.IGNORED)
        success_count = sum(1 for _, result, _ in results if result == VerifyResult.SUCCESS)
        failure_count = sum(1 for _, result, _ in results if result == VerifyResult.FAILURE)
        print(f"Ignored: {ignored_count}")
        print(f"Success: {success_count}")
        print(f"Failure: {failure_count}")
        if failure_count == 0:
            return
        
        print("\nFailures:")
        for result in [res for res in results if res[1] == VerifyResult.FAILURE]:
            print(f"{result[0]}: {result[2]}")


if __name__ == "__main__":
    """
    Usage:
        dataset.py stats   - Display dataset statistics
        dataset.py verify  - Verify dataset integrity
    """
    main()
