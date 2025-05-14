import os
import subprocess
import tempfile
import shutil
import re
from typing import Optional, NamedTuple, List
import uuid

def get_typst_code(raw_input: str) -> Optional[str]:
    """
    Extract Typst code from a markup-style code block.
    
    From raw text that contains:
    ```typst
    <returns this part>
    ```
    
    Args:
        raw_text: The raw text containing potential code blocks
        
    Returns:
        The extracted Typst code or None if no valid code block is found
    """
    if not raw_input:
        return None
    
    # Pattern to match code blocks with various fences and language specifiers
    patterns = [
        # Standard markup with explicit typst language tag
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
        matches = re.findall(pattern, raw_input)
        if matches:
            # Return the first match
            return matches[0].strip()
    
    # If no matches were found with any pattern
    return None

class TypstError(NamedTuple):
    """Represents an error from Typst rendering process."""

    line: int
    column: int
    message: str
    snippet: Optional[str] = None

    def __str__(self) -> str:
        return f"Line {self.line}, Column {self.column}: {self.message}"


class TypstRenderResult:
    """Represents the result of a Typst rendering operation."""

    def __init__(
        self,
        success: bool,
        pdf_path: Optional[str] = None,
        errors: Optional[List[TypstError]] = None,
        typst_file: Optional[str] = None,
        output: Optional[str] = None,
        error_output: Optional[str] = None,
        persistent_pdf_path: Optional[str] = None,
    ):
        self.success = success
        self.pdf_path = pdf_path
        self.errors = errors or []
        self.typst_file = typst_file
        self.output = output
        self.error_output = error_output
        self.persistent_pdf_path = persistent_pdf_path

    def __str__(self) -> str:
        if self.success:
            return f"Render successful: {self.pdf_path}"
        else:
            return f"Render failed with {len(self.errors)} errors"


class TypstRenderer:
    """Renders Typst code to PDF using the Typst CLI."""

    def __init__(self, typst_command: str = "typst"):
        """
        Initialize the renderer.

        Args:
            typst_command: Command to invoke the Typst CLI
        """
        self.typst_command = typst_command
        # self._check_typst_installation()

    def _check_typst_installation(self) -> None:
        """Check if Typst is installed and accessible."""
        try:
            result = subprocess.run(
                [self.typst_command, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Typst command failed: {result.stderr}")

            print(f"Typst installation verified: {result.stdout.strip()}")

        except FileNotFoundError:
            raise RuntimeError(
                f"Typst command '{self.typst_command}' not found. "
                "Please ensure Typst is installed and in your PATH."
            )

    def _parse_errors(self, error_output: str) -> List[TypstError]:
        """
        Parse error messages from Typst output.

        Args:
            error_output: Error output from Typst CLI

        Returns:
            List of parsed TypstError objects
        """
        errors = []

        # Pattern for error lines like: "error: at line 5, column 10: expected expression"
        error_pattern = r"error: at line (\d+), column (\d+): (.+?)(?:\n|$)"

        matches = re.finditer(error_pattern, error_output, re.MULTILINE)

        for match in matches:
            line = int(match.group(1))
            column = int(match.group(2))
            message = match.group(3).strip()

            # Try to extract the code snippet if it follows the error
            snippet = None
            snippet_pos = match.end()
            if snippet_pos < len(error_output):
                # Look for indented lines that might be code snippets
                snippet_match = re.search(
                    r"\n( +.+?)(?:\n\n|\n(?!\s)|$)",
                    error_output[snippet_pos:],
                    re.DOTALL,
                )
                if snippet_match:
                    snippet = snippet_match.group(1).strip()

            errors.append(
                TypstError(line=line, column=column, message=message, snippet=snippet)
            )

        # If regex didn't match but there are error messages, create a generic error
        if not errors and "error" in error_output.lower():
            errors.append(TypstError(line=0, column=0, message=error_output.strip()))

        return errors

    def render_file(self, typst_file, pdf_file) -> TypstRenderResult:
        try:
            process = subprocess.run(
                [self.typst_command, "compile", typst_file, pdf_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            success = process.returncode == 0

            if success:
                return TypstRenderResult(
                    success=True,
                    pdf_path=pdf_file,
                    typst_file=typst_file,
                    output=process.stdout,
                    error_output=process.stderr,
                )
            else:
                # Parse errors from output
                errors = self._parse_errors(process.stderr)

                return TypstRenderResult(
                    success=False,
                    errors=errors,
                    typst_file=typst_file,
                    output=process.stdout,
                    error_output=process.stderr,
                )

        except Exception as e:
            return TypstRenderResult(
                success=False,
                errors=[TypstError(line=0, column=0, message=str(e))],
                typst_file=typst_file,
            )

    def render(
        self,
        typst_code: str,
        typ_path: Optional[str]=None,
        pdf_path: Optional[str]=None
    ) -> TypstRenderResult:
        """
        Render Typst code to PDF.

        Args:
            typst_code: Typst code to render
            output_dir: Directory to save the output (temporary dir if None)
            filename_prefix: Prefix for generated filenames

        Returns:
            TypstRenderResult with render results
        """
        # Create a temporary directory for the output if not specified
        if not typ_path:
            typ_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".typ")
            typ_path = typ_temp_file.name

        if not pdf_path:
            pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        else:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        # Write Typst code to file
        with open(typ_path, "w", encoding="utf-8") as f:
            f.write(typst_code)

        # Run Typst compiler
        try:
            process = subprocess.run(
                [self.typst_command, "compile", typ_path, pdf_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            success = process.returncode == 0

            if success:
                return TypstRenderResult(
                    success=True,
                    pdf_path=pdf_path,
                    typst_file=typ_path,
                    output=process.stdout,
                    error_output=process.stderr,
                )
            else:
                # Parse errors from output
                errors = self._parse_errors(process.stderr)

                return TypstRenderResult(
                    success=False,
                    errors=errors,
                    typst_file=typ_path,
                    output=process.stdout,
                    error_output=process.stderr,
                )

        except Exception as e:
            return TypstRenderResult(
                success=False,
                errors=[TypstError(line=0, column=0, message=str(e))],
                typst_file=typ_path,
            )
    
    def render_sample(
        self, sample_output: str, output_dir: Optional[str] = None
    ) -> TypstRenderResult:
        """
        Render the output of a TypstBench sample.

        Args:
            sample_output: Typst code from a sample's output
            output_dir: Directory to save the output

        Returns:
            TypstRenderResult with render results
        """
        return self.render(sample_output, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Typst code to PDF")
    parser.add_argument("--code", help="Typst code to render")
    parser.add_argument("--file", help="File containing Typst code")
    parser.add_argument("--output-dir", help="Directory to save output")

    args = parser.parse_args()

    # Get Typst code from arguments
    typst_code = ""
    if args.code:
        typst_code = args.code
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            typst_code = f.read()
    else:
        print("Error: Must provide either --code or --file")
        exit(1)

    # Render Typst code
    renderer = TypstRenderer()
    result = renderer.render(typst_code, args.output_dir)

    # Print result
    if result.success:
        print(f"Successfully rendered to: {result.pdf_path}")
    else:
        print("Rendering failed with errors:")
        for error in result.errors:
            print(f"  {error}")

        print("\nError output:")
        print(result.error_output)
