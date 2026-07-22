import argparse
import os
import numpy as np
import pandas as pd

def get_scores_from_file(file_path):
    """Супер-надежная функция чтения готовых скоров из файла любого формата"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден по пути: {file_path}")
        
    if os.path.getsize(file_path) == 0:
        return np.array([])
        
    try:
        df = pd.read_csv(file_path)
        if not df.empty:
            last_col = df.columns[-1]
            scores = pd.to_numeric(df[last_col], errors='coerce').dropna().values
            if len(scores) > 0:
                return scores
    except Exception:
        pass

    try:
        scores = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().replace(',', ' ').split()
                if parts:
                    try:
                        val = float(parts[-1])
                        if 0 <= val <= 1:
                            scores.append(val)
                    except ValueError:
                        continue
        return np.array(scores)
    except Exception as e:
        raise IOError(f"Критическая ошибка при разборе файла {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Генерация сводной статистической таблицы на основе трех файлов со скорами"
    )
    parser.add_argument('-f1', '--file1', type=str, required=True, help='Путь к первому файлу')
    parser.add_argument('-f2', '--file2', type=str, required=True, help='Путь ко второму файлу')
    parser.add_argument('-f3', '--file3', type=str, required=True, help='Путь к третьему файлу')
    parser.add_argument('-o', '--output-csv', type=str, required=True, help='Путь для сохранения таблицы (.csv)')
    args = parser.parse_args()

    files = [args.file1, args.file2, args.file3]
    records = []

    for file_path in files:
        name = os.path.basename(file_path)
        scores = get_scores_from_file(file_path)
        
        if len(scores) == 0:
            records.append({
                "Метод": name, "Элементов прочитано": 0, "Среднее (Mean)": 0, 
                "Минимум": 0, "25% (Q1)": 0, "Медиана (Q2)": 0, 
                "75% (Q3)": 0, "Максимум": 0
            })
            continue

        records.append({
            "Метод": name,
            "Элементов прочитано": len(scores),
            "Среднее (Mean)": np.mean(scores),
            "Минимум": np.min(scores),
            "25% (Q1)": np.percentile(scores, 25),
            "Медиана (Q2)": np.median(scores),
            "75% (Q3)": np.percentile(scores, 75),
            "Максимум": np.max(scores)
        })

    df_report = pd.DataFrame(records)
    
    # Округление до 4 знаков после запятой
    numeric_cols = ["Среднее (Mean)", "Минимум", "25% (Q1)", "Медиана (Q2)", "75% (Q3)", "Максимум"]
    df_report[numeric_cols] = df_report[numeric_cols].round(4)

    # Создание директории, если её нет
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Сохранение таблицы в CSV
    df_report.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    
    # 2. Корректное формирование пути и сохранение в Excel (.xlsx)
    base_path, _ = os.path.splitext(args.output_csv)
    output_xlsx = base_path + '.xlsx'
    
    try:
        df_report.to_excel(output_xlsx, index=False)
        xlsx_status = f"и Excel успешно сохранены в папку: {output_dir}"
    except ImportError:
        xlsx_status = f"\n⚠️ Не удалось создать Excel, так как не установлен модуль openpyxl для Python 3."

    # Печать в консоль
    print("\n📊 СВОДНАЯ СТАТИСТИЧЕСКАЯ ТАБЛИЦА (Данные для Boxplot):")
    print(df_report.to_string(index=False))
    print(f"\n🎉 Файлы CSV {xlsx_status}")

if __name__ == "__main__":
    main()


