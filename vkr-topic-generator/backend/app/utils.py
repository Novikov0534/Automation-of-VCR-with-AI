POSITION_PREFIXES = {
    "преподаватель": "преп.",
    "доцент": "доц.",
    "профессор": "проф.",
    "преп.": "преп.",
    "доц.": "доц.",
    "проф.": "проф.",
}


def short_teacher_name(full_name: str) -> str:
    """Return Фамилия И.О. for a full Russian-style teacher name.

    The database keeps the full name. This helper is only for generated-topic
    tables and exports.
    """
    parts = [part for part in full_name.strip().split() if part]
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1][0].upper()}.{parts[2][0].upper()}."
    if len(parts) == 2:
        return f"{parts[0]} {parts[1][0].upper()}."
    return full_name.strip()


def teacher_table_name(full_name: str, position: str | None = None) -> str:
    """Format a teacher for generated tables, e.g. «проф. Иванов И.И.»."""
    short = short_teacher_name(full_name)
    normalized = (position or "").strip().lower()
    prefix = POSITION_PREFIXES.get(normalized)
    return f"{prefix} {short}" if prefix else short
