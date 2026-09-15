from io import BytesIO

import xlsxwriter

from app.services.xlsx_importer import detect_historical_topics


def make_book() -> bytes:
    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
    ws = workbook.add_worksheet("История")
    headers = ["№ проекта", "ФИО Руководителя ВКР", "Индивидуальная тема"]
    for col, value in enumerate(headers):
        ws.write(0, col, value)
    ws.write_row(1, 0, [1, "Скоробогатченко Дмитрий Анатольевич", "Разработка системы видеоаналитики спортивного матча с формированием статистики"])
    ws.write_row(2, 0, [2, "", "Разработка веб-системы контроля этапов выпускной квалификационной работы"])
    ws.write_row(3, 0, [3, "Иванов Иван Иванович", "Разработка программного комплекса семантического поиска документов"])
    workbook.close()
    return buffer.getvalue()


def test_detect_historical_topics_forward_fills_teacher_and_year():
    result = detect_historical_topics(make_book(), "ВКР_24-25.xlsx")
    assert len(result) == 1
    assert result[0].name == "История"
    assert len(result[0].items) == 3
    assert result[0].items[1].teacher_name == "Скоробогатченко Дмитрий Анатольевич"
    assert result[0].items[0].year == 2025
    assert result[0].items[2].project == "3"
