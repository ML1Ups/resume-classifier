import csv
import random
from pathlib import Path

PROFILES = {
    "backend": {
        "skills": [
            "Python",
            "FastAPI",
            "Django",
            "PostgreSQL",
            "Redis",
            "REST API",
            "SQLAlchemy",
            "asyncio",
            "RabbitMQ",
            "Docker",
            "Go",
            "gRPC",
        ],
        "tasks": [
            "Разработал API для каталога товаров и авторизацию пользователей.",
            "Оптимизировал запросы к базе и транзакции при оформлении заказов.",
            "Реализовал асинхронную обработку задач и интеграцию с платежами.",
            "Писал серверные сервисы и интеграционные тесты.",
            "Поддерживал микросервисы с очередями сообщений и кешированием.",
            "Проектировал схему данных и миграции базы.",
            "Снижал время ответа сервисов и устранял N+1 запросы.",
            "Added request validation, permissions and transaction handling.",
        ],
    },
    "frontend": {
        "skills": [
            "JavaScript",
            "TypeScript",
            "React",
            "Vue",
            "HTML",
            "CSS",
            "Redux",
            "Webpack",
            "Vite",
            "Jest",
            "REST API",
            "Playwright",
        ],
        "tasks": [
            "Разрабатывал адаптивные интерфейсы личного кабинета.",
            "Создал библиотеку компонентов и формы с валидацией.",
            "Настроил управление состоянием и взаимодействие с API.",
            "Верстал страницы по макетам и проверял доступность.",
            "Уменьшил размер клиентского бандла и ускорил загрузку страниц.",
            "Реализовал маршрутизацию и обработку ошибок в браузере.",
            "Писал компонентные тесты и исправлял визуальные регрессии.",
            "Built accessible web components and responsive dashboards.",
        ],
    },
    "data_science": {
        "skills": [
            "Python",
            "pandas",
            "NumPy",
            "scikit-learn",
            "CatBoost",
            "PyTorch",
            "MLflow",
            "SQL",
            "NLP",
            "cross-validation",
            "Optuna",
            "embeddings",
        ],
        "tasks": [
            "Обучал модели классификации и подбирал гиперпараметры.",
            "Строил признаки для прогнозирования оттока клиентов.",
            "Сравнивал модели по метрикам на отложенной выборке.",
            "Разработал ранжирование и рекомендательную систему.",
            "Исследовал переобучение, дисбаланс классов и утечку данных.",
            "Обрабатывал тексты и обучал нейросетевые модели.",
            "Настроил эксперименты и воспроизводимость обучения.",
            "Trained predictive models with feature engineering and validation.",
        ],
    },
    "qa": {
        "skills": [
            "Python",
            "pytest",
            "Selenium",
            "Playwright",
            "Postman",
            "SQL",
            "Allure",
            "Jira",
            "REST API",
            "JMeter",
            "Java",
            "Git",
        ],
        "tasks": [
            "Составлял тест-кейсы и проверял требования к продукту.",
            "Автоматизировал регрессионные проверки API и интерфейса.",
            "Находил дефекты и оформлял воспроизводимые баг-репорты.",
            "Проводил функциональное и нагрузочное тестирование.",
            "Поддерживал тестовые стенды и проверял исправления.",
            "Анализировал логи и результаты автотестов.",
            "Писал сценарии тестирования негативных случаев.",
            "Designed regression suites and verified releases against requirements.",
        ],
    },
    "devops": {
        "skills": [
            "Linux",
            "Docker",
            "Kubernetes",
            "Terraform",
            "Ansible",
            "GitLab CI",
            "Prometheus",
            "Grafana",
            "Bash",
            "Python",
            "Nginx",
            "PostgreSQL",
        ],
        "tasks": [
            "Настроил автоматическое развертывание и CI/CD пайплайны.",
            "Поддерживал кластеры и сетевую инфраструктуру.",
            "Описывал инфраструктуру как код и автоматизировал конфигурацию.",
            "Внедрил мониторинг, алерты и сбор логов сервисов.",
            "Настроил резервное копирование и восстановление после сбоев.",
            "Разбирал инциденты и устранял проблемы доступности.",
            "Управлял контейнерами и секретами в окружениях.",
            "Maintained deployment pipelines, infrastructure and service reliability.",
        ],
    },
    "data_analytics": {
        "skills": [
            "SQL",
            "Python",
            "pandas",
            "Excel",
            "Power BI",
            "Tableau",
            "A/B tests",
            "ClickHouse",
            "statistics",
            "Metabase",
            "ETL",
            "Git",
        ],
        "tasks": [
            "Создавал отчеты и дашборды для продуктовой команды.",
            "Анализировал воронку продаж и причины изменения метрик.",
            "Проводил A/B эксперименты и оценивал статистическую значимость.",
            "Строил витрины данных и проверял качество отчетности.",
            "Исследовал поведение пользователей и когортное удержание.",
            "Готовил аналитические выводы и рекомендации бизнесу.",
            "Согласовывал определения метрик и рассчитывал KPI.",
            "Analyzed business metrics and created decision-support dashboards.",
        ],
    },
}


def main() -> None:
    rng = random.Random(20261009)
    rows = []
    for category, profile in PROFILES.items():
        for index in range(40):
            skills = rng.sample(profile["skills"], rng.randint(3, 7))
            if index % 4 == 0:
                neighbor = rng.choice([k for k in PROFILES if k != category])
                skills.extend(rng.sample(PROFILES[neighbor]["skills"], 2))
            tasks = rng.sample(profile["tasks"], rng.randint(1, 3))
            experience = rng.choice(
                [
                    "Учебный проект.",
                    "Стажировка в продуктовой команде.",
                    "Опыт работы в команде разработки.",
                    "Практика при вузе.",
                ]
            )
            education = rng.choice(
                [
                    "Образование: информационные системы.",
                    "Образование: прикладная математика.",
                    "Образование: программная инженерия.",
                ]
            )
            soft = rng.choice(
                [
                    "Работал с Git и участвовал в код-ревью.",
                    "Умею обсуждать требования и документировать решения.",
                    "Работал в команде по Scrum, использовал английский язык.",
                ]
            )
            text = f"{experience} {' '.join(tasks)} Навыки: {', '.join(skills)}. {education} {soft}"
            rows.append({"text": text, "category": category})
    rng.shuffle(rows)
    output = Path(__file__).resolve().parents[1] / "data" / "resumes_demo.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "category"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
