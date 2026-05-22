from pathlib import Path


class TextParser:
    def parse(self, path: str) -> str:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if file_path.suffix.lower() != ".txt":
            raise ValueError(f"TextParser only supports .txt files, got: {file_path.suffix}")

        return file_path.read_text(encoding="utf-8")