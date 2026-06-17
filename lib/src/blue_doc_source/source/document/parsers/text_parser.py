from pathlib import Path


class TextParser:
    """Reads a UTF-8 .txt file and returns its full contents as a string."""

    def parse(self, path: str) -> str:
        """Return the full text content of a .txt file.

        Args:
            path: Absolute or relative path to a .txt file.

        Returns:
            The file's text content as a single string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .txt.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.suffix.lower() != ".txt":
            raise ValueError(f"TextParser only supports .txt files, got: {file_path.suffix}")

        return file_path.read_text(encoding="utf-8")