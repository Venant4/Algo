import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# 1. Lecture des données depuis un fichier Excel
def importer_depuis_excel(fichier, feuille=0):
    """Importe les données depuis un fichier Excel."""
    df = pd.read_excel(fichier, sheet_name=feuille)
    return df.values  # Retourne un numpy.array

# 2. Calcul manuel de la matrice de distances
def calculer_matrice_distances(X, metrique='euclidienne'):
    """Calcule la matrice des distances entre tous les points."""
    n = X.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if metrique == 'euclidienne':
                dist_matrix[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
            elif metrique == 'manhattan':
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
            dist_matrix[j, i] = dist_matrix[i, j]  # Symétrie
    return dist_matrix

# 3. Implémentation manuelle du linkage
def linkage_manuelle(dist_matrix, methode='single'):
    n = dist_matrix.shape[0]
    clusters = [[i] for i in range(n)]  # Initialisation
    Z = []
    next_cluster_idx = n  # Premier indice pour les nouveaux clusters
    
    # Dictionnaire pour suivre les indices des clusters
    cluster_indices = {i: i for i in range(n)}
    
    for k in range(n - 1):
        # Trouver la paire de clusters la plus proche
        min_dist = np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if methode == 'single':
                    dist = np.min([dist_matrix[a][b] for a in clusters[i] for b in clusters[j]])
                elif methode == 'complete':
                    dist = np.max([dist_matrix[a][b] for a in clusters[i] for b in clusters[j]])
                elif methode == 'average':
                    dist = np.mean([dist_matrix[a][b] for a in clusters[i] for b in clusters[j]])
                
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j
        
        # Fusionner les clusters
        new_cluster = clusters[best_i] + clusters[best_j]
        new_size = len(new_cluster)
        
        # Obtenir les vrais indices des clusters
        idx_i = cluster_indices[clusters[best_i][0]]
        idx_j = cluster_indices[clusters[best_j][0]]
        
        # Enregistrer la fusion
        Z.append([idx_i, idx_j, min_dist, new_size])
        
        # Mettre à jour les indices
        cluster_indices[new_cluster[0]] = next_cluster_idx
        
        # Mettre à jour la liste des clusters
        clusters.append(new_cluster)
        del clusters[best_j], clusters[best_i]
        next_cluster_idx += 1
    
    return np.array(Z)

# 4. Application complète
if __name__ == "__main__":
    # Import des données (remplacer 'donnees.xlsx' par votre fichier)
    try:
        X = importer_depuis_excel('application.xlsx')
    except FileNotFoundError:
        print("Erreur: Fichier non trouvé. Utilisation de données exemple.")
        X = np.array([[1, 2], [1, 4], [4, 2], [4, 4]])  # Données exemple

    # Paramètres
    metrique = 'euclidienne'  # ou 'manhattan'
    methode_linkage = 'average'  # 'single', 'complete', 'average'

    # Calculs
    dist_matrix = calculer_matrice_distances(X, metrique)
    Z = linkage_manuelle(dist_matrix, methode_linkage)

    # Affichage
    print("Matrice de linkage:\n", Z)
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f"CAH - Linkage {methode_linkage} / Distance {metrique}")
    plt.show()


