from pathlib import Path
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md"}


def load_file(file_path):
    """Load a single PDF, HTML, or Markdown file with rich metadata."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = []
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        try:
            documents = PyPDFLoader(str(file_path)).load()
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF file {file_path.name}: {e}")

    elif extension in {".html", ".htm"}:
        try:
            html = file_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            documents = [
                Document(
                    page_content=text,
                    metadata={"source_file": file_path.name, "page": 0}
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to load HTML file {file_path.name}: {e}")

    elif extension == ".md":
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            documents = [
                Document(
                    page_content=text,
                    metadata={"source_file": file_path.name, "page": 0}
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to load Markdown file {file_path.name}: {e}")

    else:
        return []

    # Standardize metadata across all document types
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
        try:
            doc.metadata["source_path"] = str(file_path.relative_to(BASE_DIR))
        except ValueError:
            doc.metadata["source_path"] = file_path.name
        doc.metadata["file_type"] = extension.lstrip(".")
        doc.metadata["char_count"] = len(doc.page_content)

    return documents


def load_material(data_dir=DATA_DIR):
    """Load all supported documents from the data directory."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = []
    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for file_path in files:
        loaded = load_file(file_path)
        documents.extend(loaded)

    print(f"[load_material] Files scanned: {len(files)}")
    print(f"[load_material] Document pages/sections loaded: {len(documents)}")

    return documents
