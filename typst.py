import os
import subprocess
import tempfile
import shutil
import re
from typing import Optional, NamedTuple, List
import uuid


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
        temp_dir: Optional[str] = None,
        typst_file: Optional[str] = None,
        output: Optional[str] = None,
        error_output: Optional[str] = None,
    ):
        self.success = success
        self.pdf_path = pdf_path
        self.errors = errors or []
        self.temp_dir = temp_dir
        self.typst_file = typst_file
        self.output = output
        self.error_output = error_output

    def __str__(self) -> str:
        if self.success:
            return f"Render successful: {self.pdf_path}"
        else:
            return f"Render failed with {len(self.errors)} errors"

    def cleanup(self) -> None:
        """Remove temporary files created during rendering."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TypstRenderer:
    """Renders Typst code to PDF using the Typst CLI."""

    def __init__(self, typst_command: str = "typst"):
        """
        Initialize the renderer.

        Args:
            typst_command: Command to invoke the Typst CLI
        """
        self.typst_command = typst_command
        self._check_typst_installation()

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
                    temp_dir=None,
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
                    temp_dir=None,
                    typst_file=typst_file,
                    output=process.stdout,
                    error_output=process.stderr,
                )

        except Exception as e:
            return TypstRenderResult(
                success=False,
                errors=[TypstError(line=0, column=0, message=str(e))],
                temp_dir=None,
                typst_file=typst_file,
            )

    def render(
        self,
        typst_code: str,
        output_dir: Optional[str] = None,
        filename_prefix: str = "typst_render",
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
        temp_dir_obj = None
        if not output_dir:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="typst_render_")
            output_dir = temp_dir_obj.name

        temp_dir = output_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Generate unique filenames
        unique_id = uuid.uuid4().hex[:8]
        typst_file = os.path.join(temp_dir, f"{filename_prefix}_{unique_id}.typ")
        pdf_file = os.path.join(temp_dir, f"{filename_prefix}_{unique_id}.pdf")

        # Write Typst code to file
        with open(typst_file, "w", encoding="utf-8") as f:
            f.write(typst_code)

        # Run Typst compiler
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
                    temp_dir=temp_dir,
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
                    temp_dir=temp_dir,
                    typst_file=typst_file,
                    output=process.stdout,
                    error_output=process.stderr,
                )

        except Exception as e:
            return TypstRenderResult(
                success=False,
                errors=[TypstError(line=0, column=0, message=str(e))],
                temp_dir=temp_dir,
                typst_file=typst_file,
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
    typst_code = None
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
