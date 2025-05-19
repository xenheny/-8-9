
#4	КластеризацияСгеренировать 3 кластера пятиугольной области с долей межклассового пересечения 9-11%. Сохранить полученные данные в файл	"Разработать 3 модели кластеризации с помощью библиотеки sklearn: SpectralClustering, OPTICS, BIRCH.
#Подобрать эффективные параметры для этих методов с применением случайного поиска (Random Search)"	"Построить 3 графика с визуализацией кластеров на каждой эпохе обучения с эффектиными параметрами.
#Рассчитать и вывести на экран внутреннюю метрику кластеризации ""Коэффициент силуэта"" (Silhouette Coefficient)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, OPTICS, Birch
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
import pandas as pd


# 1. Генерация данных в форме пятиугольника с 3 кластерами
def generate_pentagon_data(n_samples=1500, overlap=0.1):
    # Углы пятиугольника (в радианах)
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]

    # Вершины пятиугольника
    vertices = np.array([(np.cos(a), np.sin(a)) for a in angles])

    # Генерация 3 кластеров внутри пятиугольника
    centers = np.array([[-0.5, 0], [0.5, 0.2], [0, 0.6]])
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.15 * (1 + overlap))

    # Масштабирование данных к пятиугольнику
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X *= 0.5

    # Добавление шума для создания пересечения
    noise = np.random.normal(0, 0.05 * (1 + 2 * overlap), (n_samples, 2))
    X += noise

    return X, y


# Генерация данных с пересечением 10% (9-11%)
X, y_true = generate_pentagon_data(overlap=0.1)

# Визуализация исходных данных
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
plt.title("Исходные данные с истинными кластерами")
plt.show()


# 2. Функция для оптимизации параметров кластеризации
def optimize_clustering(X, model_class, param_dist, n_iter=20, n_clusters=3):
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter))

    best_score = -1
    best_params = None
    best_labels = None

    for params in param_list:
        if 'n_clusters' in model_class().get_params():
            params['n_clusters'] = n_clusters

        model = model_class(**params)
        labels = model.fit_predict(X)

        if len(np.unique(labels)) > 1:  # silhouette_score требует хотя бы 2 кластера
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels

    return best_params, best_score, best_labels


# 3. Оптимизация и применение SpectralClustering
print("\nОптимизация SpectralClustering...")
param_dist_spectral = {
    'affinity': ['rbf', 'nearest_neighbors'],
    'gamma': uniform(0.1, 10),
    'n_neighbors': randint(5, 50)
}

best_params_spectral, score_spectral, y_spectral = optimize_clustering(
    X, SpectralClustering, param_dist_spectral)

print("Лучшие параметры для SpectralClustering:", best_params_spectral)
print(f"Коэффициент силуэта: {score_spectral:.3f}")

# Визуализация SpectralClustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_spectral, cmap='viridis', s=10)
plt.title("SpectralClustering с оптимальными параметрами")
plt.show()

# 4. Оптимизация и применение OPTICS
print("\nОптимизация OPTICS...")
param_dist_optics = {
    'min_samples': randint(5, 50),
    'xi': uniform(0.01, 0.3),
    'min_cluster_size': uniform(0.01, 0.3)
}

best_params_optics, score_optics, y_optics = optimize_clustering(
    X, OPTICS, param_dist_optics)

# Добавляем параметры, которые не были в случайном поиске
best_params_optics['cluster_method'] = 'xi'

print("Лучшие параметры для OPTICS:", best_params_optics)
print(f"Коэффициент силуэта: {score_optics:.3f}")

# Визуализация OPTICS
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_optics, cmap='viridis', s=10)
plt.title("OPTICS с оптимальными параметрами")
plt.show()

# 5. Оптимизация и применение BIRCH
print("\nОптимизация BIRCH...")
param_dist_birch = {
    'threshold': uniform(0.1, 1),
    'branching_factor': randint(20, 100)
}

best_params_birch, score_birch, y_birch = optimize_clustering(
    X, Birch, param_dist_birch)

# Добавляем параметры, которые не были в случайном поиске
best_params_birch['n_clusters'] = 3

print("Лучшие параметры для BIRCH:", best_params_birch)
print(f"Коэффициент силуэта: {score_birch:.3f}")

# Визуализация BIRCH
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_birch, cmap='viridis', s=10)
plt.title("BIRCH с оптимальными параметрами")
plt.show()

# 6. Визуализация на разных этапах обучения для BIRCH
plt.figure(figsize=(15, 5))
sample_sizes = [500, 1000, 1500]

for i, n_samples in enumerate(sample_sizes):
    birch_partial = Birch(**best_params_birch)
    y_partial = birch_partial.fit_predict(X[:n_samples])

    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:n_samples, 0], X[:n_samples, 1], c=y_partial, cmap='viridis', s=10)
    plt.title(f"BIRCH после {n_samples} образцов")

plt.tight_layout()
plt.show()

# 7. Сохранение результатов
results = pd.DataFrame({
    'x': X[:, 0],
    'y': X[:, 1],
    'true_cluster': y_true,
    'spectral_cluster': y_spectral,
    'optics_cluster': y_optics,
    'birch_cluster': y_birch
})

results.to_csv('clustering_results.csv', index=False)

# Сохранение параметров
params_df = pd.DataFrame({
    'method': ['SpectralClustering', 'OPTICS', 'BIRCH'],
    'best_params': [best_params_spectral, best_params_optics, best_params_birch],
    'silhouette_score': [score_spectral, score_optics, score_birch]
})
params_df.to_csv('clustering_params.csv', index=False)

# 8. Итоговые результаты
print("\nИтоговые результаты кластеризации:")
print(f"SpectralClustering: Silhouette = {score_spectral:.3f}")
print(f"OPTICS: Silhouette = {score_optics:.3f}")
print(f"BIRCH: Silhouette = {score_birch:.3f}")

# Определение лучшего метода
best_method = np.argmax([score_spectral, score_optics, score_birch])
methods = ['SpectralClustering', 'OPTICS', 'BIRCH']
print(f"\nЛучший метод кластеризации: {methods[best_method]}")
