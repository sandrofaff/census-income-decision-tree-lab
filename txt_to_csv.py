import pandas as pd

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

def convert_txt_to_csv(txt_path, csv_path):
    df = pd.read_csv(
        txt_path,
        header=None,
        names=COLUMN_NAMES,
        sep=r',\s*',          # запятая + опциональные пробелы
        engine='python',
        na_values='?'
    )

    # Отбрасываем служебные строки-заголовки (например, "|1x3 Cross validator"),
    # у которых поле age не является числом.
    age_as_str = df['age'].astype(str).str.strip()
    df = df[age_as_str.str.match(r'^\d+$')].reset_index(drop=True)

    # В adult.test в конце строки бывает точка — убираем
    df['income'] = df['income'].astype(str).str.replace('.', '', regex=False).str.strip()

    df.to_csv(csv_path, index=False)
    print(f"{txt_path} -> {csv_path}  ({len(df)} строк)")


if __name__ == '__main__':
    convert_txt_to_csv('adult.data.txt', 'adult_train.csv')
    convert_txt_to_csv('adult.test.txt', 'adult_test.csv')