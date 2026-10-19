# Предобработка датасета MOF

Запуск скриптов предобработки:

python src/all_dbs_former.py — создает datasets/temp_dataset/final_master_index.csv. Значения refcode для CoRE копируются из исходных таблиц ASR/FSR/ION; refcode для QMOF считываются из валидного префикса имени до первого подчеркивания; для USMOF поле refcode остается пустым. master_id и пути к CIF для CoRE используют coreid.

python src/first_filter.py — выполняет дедупликацию по непустым REFCODE, стандартизирует структуры с помощью spglib и оставляет примитивные ячейки, содержащие не более 100 атомов (узлов). Стандартизированные CIF-файлы записываются в datasets/temp_dataset/standardized_structures/, а итоговый индекс сохраняется в first_filtered_master_index.csv.

python src/second_filter.py — использует StructureMatcher для удаления дубликатов среди стандартизированных структур. Результат записывается в second_filtered_master_index.csv, а совпавшие (дублирующиеся) строки перечисляются в second_filter_duplicates.csv.

python src/third_filter.py — запускает проверки MOFChecker для структур, оставшихся после второго фильтра. Чистый результат сохраняется в third_filtered_master_index.csv, а причины отклонения и ошибки обработки фиксируются в отдельных файлах.

python src/load.py — копирует примитивные CIF-файлы из вывода третьего фильтра в datasets/final_dataset/structures/ и записывает свойства структур в datasets/final_dataset/structures.csv. Таблица включает ID, сокращенную формулу, металлы, символ пространственной группы, сингонию (кристаллическую систему), количество атомов в примитивной ячейке, тип расчетной ячейки, источник и путь к CIF. Ошибки загрузки или анализа симметрии для отдельных строк записываются в datasets/final_dataset/load_errors.csv.