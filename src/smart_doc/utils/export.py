"""Export utilities for chat, summaries, and insights."""

from datetime import datetime


def export_chat_to_markdown(messages: list) -> str:
    """Convert chat messages to a Markdown string.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Formatted Markdown string.
    """
    lines = [
        "# Chat History - Maventrag",
        f"_Diekspor pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "---\n",
    ]
    for msg in messages:
        role_label = "Pengguna" if msg["role"] == "user" else "Asisten"
        lines.append(f"### {role_label}")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_to_markdown(title: str, content: str) -> str:
    """Export any content to a formatted Markdown string.

    Args:
        title: The document title.
        content: The content body.

    Returns:
        Formatted Markdown string.
    """
    return (
        f"# {title}\n\n"
        f"_Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        f"---\n\n{content}"
    )
