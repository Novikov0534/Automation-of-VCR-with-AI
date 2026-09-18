from types import SimpleNamespace
from zipfile import ZipFile
from io import BytesIO
from xml.etree import ElementTree as ET

from app.services.xlsx_exporter import build_topics_xlsx


def _xlsx_strings(data: bytes) -> list[str]:
    with ZipFile(BytesIO(data)) as archive:
        xml = archive.read("xl/sharedStrings.xml")
    root = ET.fromstring(xml)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return ["".join(node.itertext()) for node in root.findall("x:si", ns)]


def test_publication_xlsx_contains_only_required_columns():
    teacher = SimpleNamespace(full_name="Иванов Иван Иванович", position="доцент")
    topic = SimpleNamespace(teacher=teacher, title="Разработка веб-сервиса мониторинга оборудования")

    strings = _xlsx_strings(build_topics_xlsx([topic]))

    assert strings[:3] == ["Преподаватель", "Тема", "ФИО студента"]
    assert "Источник" not in strings
    assert "Соответствие профилю" not in strings
    assert "Техническая цель / реализация" not in strings
    assert "Контакты преподавателя" not in strings
