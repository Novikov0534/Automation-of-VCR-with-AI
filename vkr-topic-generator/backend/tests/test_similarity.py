from app.services.similarity import lexical_cosine


def test_identical_is_one():
    assert lexical_cosine("Разработка интеллектуальной системы", "Разработка интеллектуальной системы") == 1.0


def test_unrelated_is_low():
    score = lexical_cosine(
        "Разработка системы анализа изображений нейронной сетью",
        "Проектирование базы данных библиотеки",
    )
    assert score < 0.4


def test_similar_is_higher_than_unrelated():
    a = "Разработка системы классификации транспортных средств с использованием нейронных сетей"
    similar = "Разработка системы классификации автомобилей на основе нейронных сетей"
    unrelated = "Разработка веб-сервиса электронного документооборота"
    assert lexical_cosine(a, similar) > lexical_cosine(a, unrelated)


def test_hybrid_keeps_unrelated_topics_low():
    from app.services.similarity import hybrid_similarity
    unrelated = hybrid_similarity(
        "Разработка рекомендательной системы для интернет-магазина продуктов питания",
        "Разработка методов управления игровыми объектами в играх жанра метроидвания",
        0.87,
    )
    near = hybrid_similarity(
        "Разработка подсистемы анализа данных о состоянии производственной киберфизической системы",
        "Разработка подсистемы сбора и предобработки данных о состоянии производственной киберфизической системы",
        0.97,
    )
    assert unrelated < 45
    assert near >= 70


def test_hybrid_exact_duplicate_is_100():
    from app.services.similarity import hybrid_similarity
    title = "Разработка приложения для переноса стиля на изображение"
    assert hybrid_similarity(title, title, 1.0) == 100.0
