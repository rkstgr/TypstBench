import os
import glob
import yaml
import re
from typing import Dict, List, Any, Optional


class TypstBenchSample:
    """Represents a single sample from the TypstBench dataset."""

    def __init__(self,
                 id: str,
                 metadata: Dict[str, Any],
                 raw_input: str,
                 raw_ground_truth: str,
                 file_path: Optional[str] = None):
        self.id = id
        self.metadata = metadata
        self.raw_input = raw_input
        self.raw_ground_truth = raw_ground_truth
        self.file_path = file_path

    def __repr__(self) -> str:
        return f"TypstBenchSample(id='{self.id}', metadata={self.metadata})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format."""
        return {
            "id": self.id,
            "metadata": self.metadata,
            "raw_input": self.raw_input,
            "raw_output": self.raw_ground_truth,
            "file_path": self.file_path
        }
    
    def get_typst_code(self) -> Optional[str]:
        """
        Extract Typst code from a markdown-style code block.
        
        From raw text that contains:
        ```typst
        <returns this part>
        ```
        
        Args:
            raw_text: The raw text containing potential code blocks
            
        Returns:
            The extracted Typst code or None if no valid code block is found
        """
        if not self.raw_ground_truth:
            return None
        
        # Pattern to match code blocks with various fences and language specifiers
        patterns = [
            # Standard markdown with explicit typst language tag
            r'```(?:typst|typ)\s*\n([\s\S]*?)\n```',
            
            # Alternative backtick count (3 or more)
            r'````(?:typst|typ)?\s*\n([\s\S]*?)\n````',
            r'`````(?:typst|typ)?\s*\n([\s\S]*?)\n`````',
            
            # Code blocks without language specification
            r'```\s*\n([\s\S]*?)\n```',
            
            # Fallback to the whole text if no code block found
            r'^([\s\S]*)$'
        ]
        
        # Try each pattern in order of preference
        for pattern in patterns:
            import re
            matches = re.findall(pattern, self.raw_ground_truth)
            if matches:
                # Return the first match
                return matches[0].strip()
        
        # If no matches were found with any pattern
        return None
    
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


class TypstBenchDataset:
    """Loader for the TypstBench dataset."""

    def __init__(self, base_dir: str = "dataset"):
        """
        Initialize the dataset loader.

        Args:
            base_dir: Base directory containing the dataset files
        """
        self.base_dir = base_dir
        self.samples = []
        self._load_dataset()

    def _parse_sample_file(self, file_path: str) -> Optional[TypstBenchSample]:
        """
        Parse a single sample file.

        Args:
            file_path: Path to the sample file

        Returns:
            A TypstBenchSample object or None if parsing fails
        """
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

            frontmatter = yaml.safe_load(frontmatter_str)

            # Get ID from frontmatter or filename if not specified
            sample_id = frontmatter.get('id', os.path.basename(file_path).split('.')[0])

            # Get input and output
            raw_input = parts[1].strip()
            raw_output = parts[2].strip()

            return TypstBenchSample(
                id=sample_id,
                metadata=frontmatter,
                raw_input=raw_input,
                raw_ground_truth=raw_output,
                file_path=file_path
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

        print(f"Loaded {len(self.samples)} samples from {self.base_dir}")

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


if __name__ == "__main__":
    # Example usage
    dataset = TypstBenchDataset()

    # Get a specific sample
    if dataset.samples:
        sample = dataset.samples[0]
        print(f"\nSample details for {sample.id}:")
        print(f"  Metadata: {sample.metadata}")
        print(f"  Input: '{sample.raw_input}'")
        print(f"  Output: '{sample.raw_output}'")
        print(f"  File: {sample.file_path}")
