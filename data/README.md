# Настоящий датасет резюме

Источник — присланный пользователем архив cv_target.csv (1).zip. В cv_target.csv 28 935 строк и 48 колонок. Исходный файл сохраняется без изменения. Скрипт преобразования не скачивает дополнительные данные и не создаёт синтетическую разметку.

Целевая метка извлекается из professionList[*].codeProfessionalSphere. Все размеченные строки исходного набора имеют ровно одну уникальную сферу; одна строка не размечена. typicalPosition почти пустая и не используется. Профессиональные коды сохранены без переименования.

## Очистка

Удаляется 1 строка без целевой сферы, 1 558 одинаковых нормализованных текстов с одинаковой меткой и 10 строк двух слишком редких классов: Logistic — 9, Entertainment — 1. Остаётся 27 366 текстов в 34 классах. Распределение после подготовки:

| Код сферы | Резюме |
|---|---:|
| DeskWork | 3642 |
| Sales | 2636 |
| Transport | 1983 |
| NotQualification | 1714 |
| Industry | 1693 |
| Medicine | 1310 |
| BuldindRealty | 1272 |
| Safety | 1264 |
| Education | 1151 |
| Communal | 1008 |
| StateServices | 978 |
| Finances | 926 |
| Culture | 765 |
| WorkingSpecialties | 752 |
| AccountingTaxesManagement | 746 |
| ServiceMaintenance | 729 |
| InformationTechnology | 678 |
| Restaurants | 542 |
| Food | 498 |
| Management | 444 |
| HumanRecruitment | 443 |
| MechanicalEngineering | 406 |
| RootLightIndustry | 317 |
| Marketing | 216 |
| ElectricpowerIndustry | 214 |
| Jurisprudence | 174 |
| HomePersonal | 163 |
| SportsFitnessBeautySalons | 148 |
| Metallurgy | 133 |
| Forest | 94 |
| Consulting | 94 |
| ChemicalAndFuelIndustry | 81 |
| Resources | 76 |
| Agricultural | 76 |

## Признаки и разбиение

Текст собирается из positionName, hardSkills, softSkills, skills, выбранных содержательных полей workExperienceList/educationList/additionalEducationList, education и additionalInformation. Из списков используются jobTitle, demands, achievements, qualification, speciality, faculty и описания курсов. Структуры в CSV записаны как Python literals: используется безопасный ast.literal_eval, с поддержкой JSON. eval не используется.

Target, ID, candidateId, idUser, innerInfo, birthday, gender, age, названия компаний и вузов не используются как признаки. Поля заполненности нужны только EDA. ID кандидата нужен только для группировки и не входит в сохранённое представление признаков. ID строки и группы в артефактах хешируются. Текст ограничен 20 000 символами.

Группы строятся до удаления повторов по связи «тот же candidateId ИЛИ тот же нормализованный текст» с транзитивным объединением. Есть 24 226 независимых групп; крупнейшая содержит 55 подготовленных строк. Группы не пересекаются между train/validation/test. StratifiedGroupKFold, seed 42, даёт 16 332 / 5 579 / 5 455 строк. Целевые доли 60/20/20 приблизительные из-за групп.

EDA и обучение словаря используют только train. Выбор модели — по validation macro F1. Test используется только для итоговой проверки победителя. Никакого семплирования пригодных данных нет.

Ограничения: сильный дисбаланс сфер, малое число примеров некоторых направлений и возможная неоднозначность исходной разметки. Модель предсказывает сферу, указанную в экспортированных данных, и не проверяет квалификацию кандидата. Две редкие сферы не входят в поддерживаемую область предсказаний.

## Тестовая фикстура

resumes_demo.csv — прежний небольшой синтетический набор. Он остался для unit-тестов, Docker smoke и CI, где настоящий CSV не хранится в Git. По умолчанию training CLI и Compose используют настоящий cv_target.csv.
