import pytest

from docmind.core.router import detect_modality
from docmind.core.schemas import Modality


@pytest.mark.parametrize(("filename", "expected"), [("a.pdf", Modality.DOCUMENT), ("sheet.xlsx", Modality.STRUCTURED), ("voice.mp3", Modality.AUDIO), ("clip.mov", Modality.VIDEO), ("photo.PNG", Modality.IMAGE)])
def test_detects_supported_modalities(filename: str, expected: Modality) -> None:
    assert detect_modality(filename) is expected


def test_rejects_unknown_file() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        detect_modality("malware.exe")
