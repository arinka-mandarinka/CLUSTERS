"""
Скрипт для визуализации многомерных данных.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12,8]


class HighDimVisualizer:

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)



    def plot_pairplot(self, X, y, feature_names, n_features=5):

        print("\n1. Pairplot")

        n_features = min(n_features, X.shape[1])
        idx = np.random.choice(X.shape[1], n_features, replace=False)

        df = pd.DataFrame(X[:, idx], columns=[feature_names[i] for i in idx])
        df["Cluster"] = y

        sns.pairplot(
            df,
            hue="Cluster",
            palette="viridis",
            diag_kind="kde",
            plot_kws={"alpha":0.6}
        )

        plt.show()



    def plot_correlation_heatmap(self, X, feature_names):

        print("\n2. Корреляционная матрица")

        df = pd.DataFrame(X, columns=feature_names)

        corr = df.corr()

        plt.figure(figsize=(12,10))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True
        )

        plt.title("Correlation Matrix")
        plt.show()


    def plot_pca_2d_3d(self, X, y):

        print("\n3. PCA 2D и 3D")

        pca = PCA(n_components=3, random_state=self.random_state)

        X_pca = pca.fit_transform(X)

        ev = pca.explained_variance_ratio_

        fig = plt.figure(figsize=(14,6))

        ax1 = fig.add_subplot(121)

        sc = ax1.scatter(
            X_pca[:,0],
            X_pca[:,1],
            c=y,
            cmap="viridis",
            alpha=0.7
        )

        ax1.set_xlabel(f"PC1 {ev[0]:.2%}")
        ax1.set_ylabel(f"PC2 {ev[1]:.2%}")
        ax1.set_title("PCA 2D")

        ax2 = fig.add_subplot(122, projection="3d")

        ax2.scatter(
            X_pca[:,0],
            X_pca[:,1],
            X_pca[:,2],
            c=y,
            cmap="viridis",
            alpha=0.7
        )

        ax2.set_title("PCA 3D")

        plt.show()



    def plot_tsne_visualization(self, X, y):

        print("\n4. t-SNE")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=self.random_state
        )

        X_tsne = tsne.fit_transform(X)

        plt.scatter(
            X_tsne[:,0],
            X_tsne[:,1],
            c=y,
            cmap="tab20",
            alpha=0.7
        )

        plt.title("t-SNE projection")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        plt.show()



    def plot_umap_visualization(self, X, y):

        if not HAS_UMAP:
            print("UMAP не установлен")
            return

        print("\n5. UMAP")

        reducer = umap.UMAP(random_state=self.random_state)

        X_umap = reducer.fit_transform(X)

        plt.scatter(
            X_umap[:,0],
            X_umap[:,1],
            c=y,
            cmap="Spectral",
            alpha=0.7
        )

        plt.title("UMAP projection")
        plt.show()



    def plot_parallel_coordinates(self, X, y, feature_names, n_features=8):

        print("\n6. Parallel Coordinates")

        n_features = min(n_features, X.shape[1])

        idx = np.random.choice(X.shape[1], n_features, replace=False)

        df = pd.DataFrame(
            X[:,idx],
            columns=[feature_names[i] for i in idx]
        )

        df["Cluster"] = y

        scaler = MinMaxScaler()

        df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:,:-1])

        pd.plotting.parallel_coordinates(
            df.sample(min(200,len(df))),
            "Cluster"
        )

        plt.xticks(rotation=45)

        plt.show()


    def plot_radar_chart(self, X, y, feature_names, n_features=6):

        print("\n7. Radar chart")

        idx = np.random.choice(X.shape[1], n_features, replace=False)

        features = [feature_names[i] for i in idx]

        classes = np.unique(y)

        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
        angles = np.concatenate((angles,[angles[0]]))

        fig = plt.figure(figsize=(6,6))

        ax = plt.subplot(111, polar=True)

        for c in classes:

            data = X[y==c][:,idx].mean(axis=0)

            values = np.concatenate((data,[data[0]]))

            ax.plot(angles, values, label=f"Cluster {c}")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)

        plt.legend()
        plt.show()



    def plot_interactive_3d(self, X, y):

        print("\n8. Interactive 3D plot")

        pca = PCA(n_components=3)

        X_pca = pca.fit_transform(X)

        fig = go.Figure()

        fig.add_trace(

            go.Scatter3d(

                x=X_pca[:,0],
                y=X_pca[:,1],
                z=X_pca[:,2],

                mode="markers",

                marker=dict(
                    size=4,
                    color=y,
                    colorscale="Viridis",
                    opacity=0.8
                )

            )

        )

        fig.update_layout(
            title="3D PCA visualization"
        )

        fig.write_html("interactive_3d_plot.html")

        print("Интерактивный график сохранён в interactive_3d_plot.html")

    def run_full_visualization(self, X, y, feature_names):

        print("\n" + "="*60)
        print("MULTIDIMENSIONAL DATA VISUALIZATION")
        print("="*60)

        self.plot_pairplot(X,y,feature_names)

        self.plot_correlation_heatmap(X,feature_names)

        self.plot_pca_2d_3d(X,y)

        self.plot_tsne_visualization(X,y)

        if HAS_UMAP:
            self.plot_umap_visualization(X,y)

        self.plot_parallel_coordinates(X,y,feature_names)

        self.plot_radar_chart(X,y,feature_names)

        self.plot_interactive_3d(X,y)

        print("\nВизуализация завершена")