"""File handling utilities."""

import os
import tempfile


def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file to a temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Path to the temporary file.
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return tmp.name


def cleanup_file(path: str) -> None:
    """Remove a temporary file if it exists.

    Args:
        path: Path to the file to remove.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def validate_file_types(uploaded_files, supported_types: list[str]) -> list:
    """Validate uploaded file types.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        supported_types: List of supported file extensions.

    Returns:
        List of invalid files (empty if all valid).
    """
    return [f for f in uploaded_files if f.name.rsplit(".", 1)[-1].lower() not in supported_types]
