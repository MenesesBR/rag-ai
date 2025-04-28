import fitz  # PyMuPDF
import io


def extract_text_from_pdf(file_path):
    doc = fitz.open(stream=file_path, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_text_into_chunks(text, max_length=500):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Divide o texto em chunks menores."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
