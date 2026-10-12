import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo



def calcMetricas(df):
    print("Frequencia das categorias:\n", df["class"].value_counts())
    print("\nMédia Geral:\n", df.drop(columns=["class"]).mean())
    print("\nModa Geral:\n", df.drop(columns=["class"]).mode())
    print("\nDesvio Padrão Geral:\n", df.drop(columns=["class"]).std())
    print("\nMédia por Categoria:\n", df.groupby("class").mean())
    print("\nDesvio Pdrão por Categoria:\n", df.groupby("class").std())



def plotaRelacoes(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            data=df,
            x="sepal length",
            y="sepal width",
            hue="class",
            palette="deep",
            s=70,
            )
    plt.title("Relação Comprimento e Largura da Sépala")
    plt.xlabel("Comprimento (cm)")
    plt.ylabel("Largura (cm)")
    plt.savefig("relacao_sepalas.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
            data=df,
            x="petal length",
            y="petal width",
            hue="class",
            palette="deep",
            s=70,
            )
    plt.title("Relação Comprimento e Largura da Pétala")
    plt.xlabel("Comprimento (cm)")
    plt.ylabel("Largura (cm)")
    plt.savefig("relacao_petalas.png", dpi=300)
    plt.close()



def plotaDistribuicao(df):
    sns.set_theme(style="whitegrid")
    fig = sns.pairplot(
            df,
            hue="class",
            palette="deep",
            diag_kind="kde",
            plot_kws={"alpha":0.8, "s":45, "edgecolor":"w", "linewidth":0.5},
            )
    fig.figure.suptitle(
            "Distribuição das Categoria sobre as 4 Dimensões",
            y=1.02,
            fontsize=14,
            weight="bold",
            )
    fig.savefig("distribuicao_multidimensional.png", dpi=300)
    plt.close()



def main():
    iris = fetch_ucirepo(id=53)
    df = iris.data.original.copy()
    calcMetricas(df)
    plotaRelacoes(df)
    plotaDistribuicao(df)


if __name__ == "__main__":
    main()
