import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def calcMetricas(df):
    print("--- Estatísticas Descritivas ---")
    print("Frequência das categorias:\n", df["class"].value_counts())
    print("\nMédia Geral:\n", df.drop(columns=["class"]).mean())
    print("\nModa Geral:\n", df.drop(columns=["class"]).mode().iloc[0])
    print("\nDesvio Padrão Geral:\n", df.drop(columns=["class"]).std())
    print("\nMédia por Categoria:\n", df.groupby("class").mean())
    print("\nModa por Categoria:\n", df.groupby("class").agg(lambda x: x.mode().iloc[0]))
    print("\nDesvio Padrão por Categoria:\n", df.groupby("class").std())
    print("-" * 30 + "\n")

def plotaRelacoes(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            data=df, x="sepal length", y="sepal width", hue="class", palette="deep", s=70
    )
    plt.title("Relação Comprimento e Largura da Sépala")
    plt.xlabel("Comprimento (cm)")
    plt.ylabel("Largura (cm)")
    plt.savefig("relacao_sepalas.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            data=df, x="petal length", y="petal width", hue="class", palette="deep", s=70
    )
    plt.title("Relação Comprimento e Largura da Pétala")
    plt.xlabel("Comprimento (cm)")
    plt.ylabel("Largura (cm)")
    plt.savefig("relacao_petalas.png", dpi=300)
    plt.close()

def plotaDistribuicao(df):
    sns.set_theme(style="whitegrid")
    fig = sns.pairplot(
            df, hue="class", palette="deep", diag_kind="kde",
            plot_kws={"alpha":0.8, "s":45, "edgecolor":"w", "linewidth":0.5}
    )
    fig.figure.suptitle(
            "Distribuição das Categorias sobre as 4 Dimensões",
            y=1.02, fontsize=14, weight="bold"
    )
    fig.savefig("distribuicao_multidimensional.png", dpi=300)
    plt.close()

def aplicaPCA(df):
    X = df.drop(columns=["class"])
    y = df["class"]
    
    # Padronização para o PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    df_pca['class'] = y.values
    
    print(f"Variância explicada pelos 2 componentes: {sum(pca.explained_variance_ratio_):.2%}\n")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", hue="class", palette="deep", s=70
    )
    plt.title("Análise de Componentes Principais (PCA) - 2D")
    plt.xlabel(f"Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.savefig("pca_iris.png", dpi=300)
    plt.close()

def classificaKNN(df):
    X = df.drop(columns=["class"])
    y = df["class"]
    
    # Divisão 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Padronização para o K-NN (usando apenas dados de treino para fit)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # K-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    print("--- Resultados da Classificação K-NN ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

def main():
    iris = fetch_ucirepo(id=53)
    df = iris.data.original.copy()
    
    calcMetricas(df)
    plotaRelacoes(df)
    plotaDistribuicao(df)
    aplicaPCA(df)
    classificaKNN(df)

if __name__ == "__main__":
    main()