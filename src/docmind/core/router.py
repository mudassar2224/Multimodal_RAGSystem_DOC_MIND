from pathlib import Path

from docmind.core.schemas import Modality

EXTENSIONS: dict[Modality, set[str]] = {
    Modality.DOCUMENT: {".pdf", ".docx", ".pptx"},
    Modality.TEXT: {".txt", ".md"},
    Modality.STRUCTURED: {".csv", ".json", ".xlsx"},
    Modality.IMAGE: {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"},
    Modality.AUDIO: {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"},
    Modality.VIDEO: {".mp4", ".mov", ".mkv", ".webm", ".avi"},
}


def detect_modality(filename: str) -> Modality:
    suffix = Path(filename).suffix.lower()
    for modality, extensions in EXTENSIONS.items():
        if suffix in extensions:
            return modality
    raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")
