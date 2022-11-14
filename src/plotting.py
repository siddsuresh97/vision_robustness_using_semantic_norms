#visualize the mds as a tree plot and add leuven_mds index as the labels
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import maxdists


def plot_dendogram(embedding_matrix, labels, title = 'Hierarchical Clustering Dendrogram', figsize=(100, 10)):
    Z = linkage(embedding_matrix, 'ward')
    plt.figure(figsize = figsize)
    plt.title(title)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=11.,  # font size for the x axis labels
        labels=labels
    )
    plt.show()

