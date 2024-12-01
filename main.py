import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed
import logging  # Импортируем модуль для логирования

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
logging.info("Загрузка данных...")
data_path = 'C:/Users/User/Desktop/rinhack/dataset'
data = pd.read_csv(data_path)
logging.info(f"Данные загружены, размер: {data.shape}")

# Первичный анализ данных (EDM)
logging.info("Первичный анализ данных (EDM)...")

# 1. Проверка на пропущенные значения
logging.info("Проверка на пропущенные значения...")
missing_values = data.isnull().sum()
logging.info(f"Количество пропущенных значений в каждом столбце:\n{missing_values}")

# 2. Статистические характеристики числовых признаков
logging.info("Статистические характеристики числовых признаков...")
desc_stats = data.describe()
logging.info(f"Статистические характеристики:\n{desc_stats}")

# 3. Распределение числовых признаков
logging.info("Построение гистограмм для числовых признаков...")
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns].hist(bins=30, figsize=(12, 10))
plt.suptitle("Распределение числовых признаков")
plt.show()

# Преобразование категориальных данных
logging.info("Преобразование категориальных данных...")
le = LabelEncoder()
for col in ['device_type', 'tran_code', 'card_type', 'oper_type', 'card_status']:
    data[col] = le.fit_transform(data[col].astype(str))
logging.info("Категориальные данные преобразованы.")

# Создание временных признаков
logging.info("Создание временных признаков...")
data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek
logging.info("Временные признаки созданы.")

# Нормализация числовых данных
logging.info("Нормализация числовых данных...")
scaler = MinMaxScaler()
data[['sum', 'balance', 'pin_inc_count']] = scaler.fit_transform(data[['sum', 'balance', 'pin_inc_count']])
logging.info("Числовые данные нормализованы.")

# Признаки для модели
features = ['sum', 'balance', 'pin_inc_count', 'device_type', 'hour',
            'day_of_week', 'tran_code', 'oper_type', 'card_status']
X = data[features]

# Разделение данных на обучающую и тестовую выборки
logging.info("Разделение данных на обучающую и тестовую выборки...")
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
logging.info("Данные разделены на обучающую и тестовую выборки.")

# Функция для обучения модели KNN и оценки F1-метрики
def train_knn(n_neighbors, metric):
    logging.info(f"Обучение модели KNN с параметрами: n_neighbors={n_neighbors}, metric={metric}")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X_train)
    
    distances, _ = knn.kneighbors(X_train)
    threshold = 0.8 * distances.max()  # Порог для аномалии на основе максимального расстояния
    
    y_train_pred = (distances[:, -1] > threshold)
    
    f1 = f1_score(y_train_pred, y_train_pred)
    
    logging.info(f"F1-метрика для KNN: {f1}")
    
    return f1, knn, n_neighbors, metric

# Параллельное обучение моделей KNN
logging.info("Запуск параллельного обучения моделей KNN...")
results = Parallel(n_jobs=-1)(delayed(train_knn)(n_neighbors, metric) 
                              for n_neighbors in [5, 10, 15, 20, 25, 30, 40, 50] 
                              for metric in ['euclidean', 'manhattan', 'chebyshev', 'cosine'])

logging.info("Параллельное обучение завершено.")

# Выбираем лучшую модель по F1-метрике
logging.info("Выбор лучшей модели KNN по F1-метрике...")
best_f1_score = 0
best_knn_model = None

for f1, knn, n_neighbors, metric in results:
    if f1 > best_f1_score:
        best_f1_score = f1
        best_knn_model = knn
        best_params = {'n_neighbors': n_neighbors, 'metric': metric}

logging.info(f"Лучшие параметры для KNN: {best_params}")

# Оценка метрик с лучшей моделью
logging.info("Оценка модели с лучшими параметрами...")
distances, _ = best_knn_model.kneighbors(X_train)
threshold = 0.8 * distances.max()
y_train_pred = (distances[:, -1] > threshold)
y_test_pred = (best_knn_model.kneighbors(X_test)[0][:, -1] > threshold)

# Определение моделей для голосования
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Создание VotingClassifier (мягкое голосование)
voting_clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')

# Обучение на тренировочных данных с метками
logging.info("Обучение VotingClassifier...")
voting_clf.fit(X_train, y_train_pred)

# Получение предсказаний
y_train_pred = voting_clf.predict(X_train)
y_test_pred = voting_clf.predict(X_test)

# Оценка метрик
logging.info(f"F1-Score (Train): {f1_score(y_train_pred, y_train_pred)}")
logging.info(f"F1-Score (Test): {f1_score(y_test_pred, y_test_pred)}")
logging.info(f"Classification Report (Test):\n{classification_report(y_test_pred, y_test_pred)}")

# Визуализация 1: Аномалии по сумме и балансу
logging.info("Построение визуализации: Аномалии по сумме и балансу...")
data['anomaly_flag'] = voting_clf.predict(X[features])  # Метки аномалий для всех данных
plt.figure(figsize=(10, 6))

sns.scatterplot(data=data, x='sum', y='balance', hue='anomaly_flag', palette={True: 'red', False: 'blue'}, legend=False)
plt.title("Аномалии по сумме и балансу")
plt.xlabel("Сумма")
plt.ylabel("Баланс")
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Аномалии'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Нормальные')]
plt.legend(handles=legend_elements, title='Аномалия', loc='upper right')
plt.show()

# Визуализация 2: Корреляционная матрица
logging.info("Построение визуализации: Корреляционная матрица...")
plt.figure(figsize=(12, 8))
sns.heatmap(data[features + ['anomaly_flag']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция признаков с аномалиями")
plt.show()

# Сохранение данных без строки заголовка
logging.info("Сохранение результатов в файл...")
output_preds_path = r'C:/Users/User/Desktop/rinhack/preds.csv'
directory = os.path.dirname(output_preds_path)

# Проверка пути и создание директории, если её нет
os.makedirs(directory, exist_ok=True)

data[['anomaly_flag']].to_csv(output_preds_path, index=False, header=False)
logging.info(f"Результаты успешно сохранены в файл: {output_preds_path}")
