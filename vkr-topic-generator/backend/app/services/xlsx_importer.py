from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import zipfile
import xml.etree.ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def _col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0
    value = 0
    for ch in letters.group(0):
        value = value * 26 + (ord(ch) - 64)
    return value - 1


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{_tag(MAIN_NS, 't')}"))
    value_node = cell.find(_tag(MAIN_NS, "v"))
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def read_xlsx_rows(content: bytes) -> dict[str, list[list[object]]]:
    """Minimal XLSX value reader implemented with the standard library.

    We only need worksheet values for import preview; formulas/styles are deliberately ignored.
    """
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("Файл не похож на корректный XLSX") from exc

    with archive:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
            raise ValueError("В XLSX не найдена структура книги")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(_tag(MAIN_NS, "si")):
                shared_strings.append("".join(node.text or "" for node in item.findall(f".//{_tag(MAIN_NS, 't')}")))

        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {
            node.attrib.get("Id", ""): node.attrib.get("Target", "")
            for node in rels_root.findall(_tag(PKG_REL_NS, "Relationship"))
        }
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets: dict[str, list[list[object]]] = {}

        for sheet_node in workbook_root.findall(f".//{_tag(MAIN_NS, 'sheet')}"):
            name = sheet_node.attrib.get("name", "Лист")
            rel_id = sheet_node.attrib.get(_tag(REL_NS, "id"), "")
            target = rels.get(rel_id)
            if not target:
                continue
            path = target.lstrip("/")
            if not path.startswith("xl/"):
                path = "xl/" + path
            if path not in names:
                continue
            sheet_root = ET.fromstring(archive.read(path))
            rows: list[list[object]] = []
            for row_node in sheet_root.findall(f".//{_tag(MAIN_NS, 'sheetData')}/{_tag(MAIN_NS, 'row')}"):
                values: dict[int, object] = {}
                max_col = -1
                for cell in row_node.findall(_tag(MAIN_NS, "c")):
                    idx = _col_index(cell.attrib.get("r", "A1"))
                    values[idx] = _cell_text(cell, shared_strings)
                    max_col = max(max_col, idx)
                if max_col < 0:
                    rows.append([])
                else:
                    rows.append([values.get(i) for i in range(max_col + 1)])
            sheets[name] = rows
        return sheets


def _norm_header(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("ё", "е"))


def _clean_teacher(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    # Контакты в примерах часто лежат в той же ячейке после переноса строки.
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(?:сф\s*[,.:;-]?\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:проф\.?|доц\.?|преп\.?)\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .;,:")


def _clean_title(value: object) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n.;")
    if text.lower() in {"-", "нет", "н/д", "?"}:
        return ""
    return text[:500]


def _infer_year(filename: str) -> int | None:
    years = [int(x) for x in re.findall(r"20\d{2}", filename)]
    if years:
        return max(years)
    match = re.search(r"(?<!\d)(\d{2})\s*[-–—_]\s*(\d{2})(?!\d)", filename)
    if match:
        right = int(match.group(2))
        return 2000 + right
    return None


@dataclass
class DetectedTopic:
    sheet: str
    row: int
    teacher_name: str
    title: str
    year: int | None
    project: str | None


@dataclass
class DetectedSheet:
    name: str
    items: list[DetectedTopic]


TITLE_PRIORITIES = (
    (100, lambda h: "индивидуальн" in h and "тем" in h),
    (98, lambda h: "тема вкр" in h or "тема выпуск" in h),
    (95, lambda h: h == "тема" or h.startswith("тема ")),
    (75, lambda h: "наименование проекта" in h),
    (70, lambda h: "тема проекта" in h),
)


def _find_header(rows: list[list[object]]) -> tuple[int, list[str]] | None:
    best: tuple[int, int, list[str]] | None = None
    for idx, row in enumerate(rows[:15]):
        headers = [_norm_header(v) for v in row]
        teacher_hits = sum(1 for h in headers if "руководител" in h)
        title_hits = 0
        for h in headers:
            if any(predicate(h) for _, predicate in TITLE_PRIORITIES):
                title_hits += 1
        score = teacher_hits * 5 + title_hits * 3
        if teacher_hits and title_hits and (best is None or score > best[0]):
            best = (score, idx, headers)
    return (best[1], best[2]) if best else None


def detect_historical_topics(content: bytes, filename: str) -> list[DetectedSheet]:
    workbook = read_xlsx_rows(content)
    year = _infer_year(filename)
    result: list[DetectedSheet] = []

    for sheet_name, rows in workbook.items():
        header_info = _find_header(rows)
        if not header_info:
            continue
        header_idx, headers = header_info
        teacher_cols = [i for i, h in enumerate(headers) if "руководител" in h]
        if not teacher_cols:
            continue
        teacher_col = teacher_cols[-1]

        title_cols: list[tuple[int, int]] = []
        for i, h in enumerate(headers):
            for priority, predicate in TITLE_PRIORITIES:
                if predicate(h):
                    title_cols.append((priority, i))
                    break
        title_cols.sort(reverse=True)
        if not title_cols:
            continue

        project_col = next((i for i, h in enumerate(headers) if h in {"№ проекта", "номер проекта"}), None)
        current_teacher = ""
        items: list[DetectedTopic] = []
        seen: set[tuple[str, str]] = set()
        for row_idx, row in enumerate(rows[header_idx + 1 :], start=header_idx + 2):
            raw_teacher = row[teacher_col] if teacher_col < len(row) else None
            teacher = _clean_teacher(raw_teacher)
            if teacher:
                current_teacher = teacher
            elif current_teacher:
                teacher = current_teacher
            if not teacher:
                continue

            title = ""
            for _, col in title_cols:
                if col < len(row):
                    title = _clean_title(row[col])
                    if title:
                        break
            if len(title) < 8:
                continue

            project = None
            if project_col is not None and project_col < len(row) and row[project_col] not in (None, ""):
                project = str(row[project_col]).strip()

            key = (teacher.casefold(), re.sub(r"\s+", " ", title.casefold()))
            if key in seen:
                continue
            seen.add(key)
            items.append(DetectedTopic(sheet_name, row_idx, teacher, title, year, project))

        if items:
            result.append(DetectedSheet(sheet_name, items))
    return result
