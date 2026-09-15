from io import BytesIO

import xlsxwriter

from ..utils import teacher_table_name


def build_topics_xlsx(topics) -> bytes:
    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
    worksheet = workbook.add_worksheet("Темы ВКР")

    header = workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    top = workbook.add_format({"valign": "top"})

    headers = ["Преподаватель", "Тема", "ФИО студента", "Контакты преподавателя"]
    for col, value in enumerate(headers):
        worksheet.write(0, col, value, header)

    for row, topic in enumerate(topics, start=1):
        worksheet.write(row, 0, teacher_table_name(topic.teacher.full_name, topic.teacher.position), top)
        worksheet.write(row, 1, topic.title, wrap)
        worksheet.write(row, 2, "", top)
        worksheet.write(row, 3, topic.teacher.contact_text, wrap)

    worksheet.set_column("A:A", 28)
    worksheet.set_column("B:B", 85)
    worksheet.set_column("C:C", 28)
    worksheet.set_column("D:D", 38)
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(1, len(topics)), len(headers) - 1)
    workbook.close()
    return buffer.getvalue()
