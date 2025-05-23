import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        """Initialise les centroïdes aléatoirement"""
        np.random.seed(self.random_state)
        indices = np.random.permutation(X.shape[0])[:self.n_clusters]
        return X[indices]
    
    def _compute_distances(self, X, centroids):
        """Calcule les distances euclidiennes au carré"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:,i] = np.sum((X - centroids[i])**2, axis=1)
        return distances
    
    def fit(self, X):
        """Entraîne le modèle K-means"""
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Étape d'assignation
            distances = self._compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Étape de mise à jour
            new_centroids = np.array([X[self.labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Vérifier la convergence
            if np.allclose(new_centroids, self.centroids):
                break
                
            self.centroids = new_centroids
    
    def predict(self, X):
        """Prédit les clusters pour de nouvelles données"""
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# 1. Chargement des données depuis Excel
def load_data_from_excel(file_path, sheet_name=0, columns=None):
    """Charge les données depuis un fichier Excel"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if columns:
        return df[columns].values
    return df.values

# 2. Application
if __name__ == "__main__":
    # Paramètres
    excel_file = "application.xlsx"  # Remplacez par votre fichier
    sheet_name = "Feuil1"        # Nom de la feuille
    columns = ["V1", "V2"]   # Colonnes à utiliser
    
    # Chargement des données
    try:
        X = load_data_from_excel("application.xlsx", "Feuil1", ["V1", "V2"] )
        print(f"Données chargées : {X.shape[0]} échantillons, {X.shape[1]} features")
        
        # Standardisation des données (recommandé pour K-means)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Application de K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # Visualisation (pour 2D)
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(X[:,0], X[:,1], c=kmeans.labels, cmap='viridis')
            plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], 
                        c='red', marker='X', s=200, alpha=0.8)
            plt.title("Résultats du K-means")
            plt.xlabel(columns[0] if columns else "Feature 1")
            plt.ylabel(columns[1] if columns else "Feature 2")
            plt.show()
        
        # Affichage des résultats
        print("\nCentroïdes finaux :")
        for i, centroid in enumerate(kmeans.centroids):
            print(f"Cluster {i}: {centroid}")
            
        # Calcul de l'inertie
        inertia = sum(np.min(kmeans._compute_distances(X, kmeans.centroids), axis=1))
        print(f"\nInertie finale: {inertia:.2f}")
        
    except FileNotFoundError:
        print(f"Erreur: Fichier {excel_file} non trouvé")
    except Exception as e:
        print(f"Erreur: {str(e)}")