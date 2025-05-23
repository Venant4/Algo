import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def normalize_data(X):
    """Normalisation manuelle Z-score"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)

def calculate_distance_matrix(X, metric):
    """Calcule la matrice des distances"""
    if metric == 'euclidean':
        return squareform(pdist(X, 'euclidean'))
    elif metric == 'cityblock':
        return squareform(pdist(X, 'cityblock'))
    else:
        return squareform(pdist(X, 'euclidean'))

def main():
    st.set_page_config(layout="wide")
    st.title("Bienvenue dans L'interface de Clustering")
    
    # Upload de fichier
    uploaded_file = st.file_uploader("📤 Importer un fichier (CSV ou Excel)", 
                                   type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            # Lecture des données
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            if df.empty:
                st.warning("Le fichier est vide ou ne contient pas de données valides")
                return
                
            # Sélection des colonnes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.error("Aucune colonne numérique trouvée dans le fichier!")
                return
                
            selected_cols = st.multiselect("🔍 Variables à utiliser", 
                                         numeric_cols,
                                         default=numeric_cols[:2])
            
            if len(selected_cols) < 2:
                st.warning("⚠️ Sélectionnez au moins 2 variables")
                return
                
            X = df[selected_cols].values
            
            # Normalisation
            normalize = st.checkbox("📏 Normaliser les données", True)
            if normalize:
                X = normalize_data(X)
            
            # Paramètres
            st.sidebar.header("Options")
            algo = st.sidebar.radio("Méthode de classification", ["CAH", "K-means"])
            metric = st.sidebar.selectbox("choix de distance", 
                                        ['euclidean', 'cityblock'])
            
            # Calcul de la matrice des distances
            distance_matrix = calculate_distance_matrix(X, metric)
            
            # Affichage de la matrice des distances
            with st.expander("🔢 Matrice des distances", expanded=False):
                st.write(pd.DataFrame(distance_matrix, 
                                    index=df.index, 
                                    columns=df.index).style.format("{:.2f}"))
            
            n_clusters = st.sidebar.slider(
                "Nombre de clusters", 
                min_value=2, 
                max_value=min(10, len(X)-1), 
                value=3
            )
            
            if algo == "CAH":
                method = st.sidebar.selectbox(
                    "Méthode de linkage",
                    ['ward', 'single', 'complete', 'average']
                )
                
                # Calcul CAH
                Z = linkage(X, method=method, metric=metric)
                clusters = fcluster(Z, n_clusters, criterion='maxclust')
                
                # Visualisation
                st.subheader("📊 Dendrogramme CAH")
                fig, ax = plt.subplots(figsize=(10, 4))
                dendrogram(Z, color_threshold=Z[-n_clusters+1, 2], ax=ax)
                st.pyplot(fig)
            
            else:  # K-means
                # Implémentation manuelle
                np.random.seed(42)
                centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]
                
                for _ in range(100):  # 100 itérations max
                    distances = cdist(X, centroids, metric)
                    clusters = np.argmin(distances, axis=1)
                    new_centroids = np.array([X[clusters == k].mean(axis=0) for k in range(n_clusters)])
                    
                    if np.all(centroids == new_centroids):
                        break
                    centroids = new_centroids
                
                # Visualisation
                st.subheader("📈 Résultats K-means")
                fig, ax = plt.subplots()
                
                if len(selected_cols) == 2:
                    ax.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', alpha=0.6)
                    ax.scatter(centroids[:,0], centroids[:,1], c='red', marker='X', s=200)
                    ax.set_xlabel(selected_cols[0])
                    ax.set_ylabel(selected_cols[1])
                else:
                    # Visualisation simplifiée pour >2 dimensions
                    for k in range(n_clusters):
                        ax.scatter(X[clusters == k, 0], X[clusters == k, 1], label=f'Cluster {k+1}')
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.legend()
                
                st.pyplot(fig)
            
            
            # Export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Exporter les résultats",
                csv,
                "clusters_results.csv",
                "text/csv",
                key='download-csv'
            )
            
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    main()