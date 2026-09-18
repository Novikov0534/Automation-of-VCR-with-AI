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


def test_lexical_fallback_catches_real_batch14_paraphrases():
    from app.services.similarity import lexical_fallback_similarity, LEXICAL_REVIEW_THRESHOLD

    pairs = [
        (
            "Многоагентная система глубинного изучения математических дисциплин с автогенерацией персонализированных заданий",
            "Разработка системы автоматической генерации и проверки учебных заданий с адаптацией сложности",
        ),
        (
            "Многоагентная система фиксации и управления историей разработки учебных и научных работ",
            "Разработка системы управления и анализа истории разработки научных проектов вуза с визуализацией зависимостей",
        ),
        (
            "Разработка системы обнаружения объектов на строительной площадке по фотографиям с формированием визуального протокола",
            "Разработка системы оценки качества строительных швов с использованием компьютерного зрения и генерации отчетов контроля с сегментацией дефектов",
        ),
    ]
    assert all(lexical_fallback_similarity(a, b) >= LEXICAL_REVIEW_THRESHOLD for a, b in pairs)


def test_lexical_fallback_keeps_different_cv_topics_apart():
    from app.services.similarity import lexical_fallback_similarity, LEXICAL_REVIEW_THRESHOLD

    score = lexical_fallback_similarity(
        "Разработка системы мониторинга заполненности мусорных контейнеров по фотографиям",
        "Разработка системы анализа спутниковых снимков для обнаружения изменений территории",
    )
    assert score < LEXICAL_REVIEW_THRESHOLD
