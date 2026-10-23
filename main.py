import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ---------- Справочник категорий (из adult.names) ----------
CATEGORICAL_VALUES = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                  'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                  '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                       'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                   'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                   'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                   'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                   'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                     'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex': ['Female', 'Male'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                       'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                       'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
                       'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                       'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
                       'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary',
                       'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                       'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                       'Holand-Netherlands'],
}

# Эталонные значения accuracy из adult.names (после удаления unknown)
REFERENCE_ACCURACY = {'C4.5': 0.8446, 'NBTree': 0.8590, 'Naive-Bayes': 0.8388}


# ---------- 1. Загрузка CSV ----------
def load_csv(path):
    return pd.read_csv(path)


# ---------- 2. Предобработка ----------
def preprocess(df):
    df = df.copy()
    # Приводим категориальные признаки к известному словарю;
    # всё, что не входит в adult.names (включая '?'), -> 'Unknown'
    for col, allowed in CATEGORICAL_VALUES.items():
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].where(df[col].isin(allowed), 'Unknown')
    # Кодируем категориальные признаки
    for col in CATEGORICAL_VALUES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # Целевая переменная
    df['income'] = df['income'].astype(str).str.strip()
    df['income'] = df['income'].map({' >50K': 1, ' <=50K': 0, '>50K': 1, '<=50K': 0})
    return df.dropna(subset=['income'])


# ---------- 3. Обучение и оценка ----------
def train_and_evaluate(X_train, y_train, X_test, y_test, criterion):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }, clf


# ---------- 4. Визуализация дерева ----------
def visualize_tree(clf, feature_names, criterion, out_path):
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=['<=50K', '>50K'],
              filled=True, rounded=True, fontsize=8, max_depth=3)
    plt.title(f'Decision Tree (criterion={criterion})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- 5. Эксперименты с долей обучающей выборки ----------
def run_split_experiments(df, criterion, ratios):
    results = []
    X = df.drop('income', axis=1)
    y = df['income']
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ratio, random_state=42, stratify=y)
        metrics, _ = train_and_evaluate(X_train, y_train, X_test, y_test, criterion)
        metrics['train_ratio'] = ratio
        results.append(metrics)
    return pd.DataFrame(results)


# ---------- 6. График метрик ----------
def plot_metrics(results_df, out_path):
    plt.figure(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(results_df['train_ratio'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Доля обучающей выборки')
    plt.ylabel('Значение метрики')
    plt.title('Зависимость метрик от доли обучающей выборки')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Путь к CSV обучающей выборки')
    parser.add_argument('--test', required=True, help='Путь к CSV тестовой выборки')
    parser.add_argument('--criterion', required=True, choices=['entropy', 'gini'],
                        help='entropy = Information Gain, gini = Gini index')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Доля обучающей выборки (1.0 = 100%%)')
    args = parser.parse_args()

    df_train = preprocess(load_csv(args.train))
    df_test = preprocess(load_csv(args.test))
    df_all = pd.concat([df_train, df_test], ignore_index=True).dropna(subset=['income'])

    X = df_all.drop('income', axis=1)
    y = df_all['income']
    print(f"Всего объектов: {len(df_all)}, доля >50K: {y.mean():.4f} "
          f"(в adult.names ожидается ~0.24)")

    if args.train_ratio < 1.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.train_ratio, random_state=42, stratify=y)
    else:
        X_train = df_train.drop('income', axis=1)
        y_train = df_train['income']
        X_test = df_test.drop('income', axis=1)
        y_test = df_test['income']

    metrics, clf = train_and_evaluate(X_train, y_train, X_test, y_test, args.criterion)

    print(f"\nКритерий: {args.criterion}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    print("\nСравнение с эталонными значениями accuracy из adult.names "
          "(после удаления unknown):")
    for name, acc in REFERENCE_ACCURACY.items():
        print(f"  {name:12s}: {acc:.4f}  (разница: {metrics['accuracy'] - acc:+.4f})")

    os.makedirs('results', exist_ok=True)
    tree_path = f"results/tree_{args.criterion}.png"
    visualize_tree(clf, X.columns, args.criterion, tree_path)
    print(f"\nДерево сохранено: {tree_path}")

    ratios = [0.6, 0.7, 0.8, 0.9]
    results_df = run_split_experiments(df_all, args.criterion, ratios)
    print("\nРезультаты экспериментов (доля обучающей выборки -> метрики):")
    print(results_df.to_string(index=False))

    plot_metrics(results_df, f"results/metrics_vs_split_{args.criterion}.png")
    print(f"График метрик сохранён: results/metrics_vs_split_{args.criterion}.png")


if __name__ == '__main__':
    main()