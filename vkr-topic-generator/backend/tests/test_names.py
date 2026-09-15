from app.utils import short_teacher_name, teacher_table_name


def test_short_teacher_name_full_fio():
    assert short_teacher_name("Скоробогаченко Дмитрий Анатольевич") == "Скоробогаченко Д.А."


def test_short_teacher_name_two_parts_fallback():
    assert short_teacher_name("Иванов Иван") == "Иванов И."


def test_teacher_table_name_with_position_prefixes():
    assert teacher_table_name("Скоробогаченко Дмитрий Анатольевич", "профессор") == "проф. Скоробогаченко Д.А."
    assert teacher_table_name("Иванов Иван Иванович", "доцент") == "доц. Иванов И.И."
    assert teacher_table_name("Петров Петр Петрович", "преподаватель") == "преп. Петров П.П."


def test_teacher_table_name_without_position():
    assert teacher_table_name("Иванов Иван Иванович", None) == "Иванов И.И."
