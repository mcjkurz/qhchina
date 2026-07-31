import numpy as np
from typing import Any
from ._vector_ops import (
    cosine_similarity,
    cosine_distance,
    most_similar,
    align_vectors,
)


__all__ = [
    'project_2d',
    'get_bias_direction',
    'calculate_bias',
    'project_bias',
    'cosine_similarity',
    'cosine_distance',
    'most_similar',
    'align_vectors',
]


def project_2d(
    vectors: list[np.ndarray] | dict[str, np.ndarray] | np.ndarray, 
    labels: list[str] | None = None, 
    method: str = 'pca', 
    title: str | None = None, 
    color: str | list[str] | None = None, 
    figsize: tuple[int, int] = (8, 8), 
    fontsize: int = 12, 
    perplexity: float | None = None,
    filename: str | None = None,
    adjust_text_labels: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> None:
    """
    Projects high-dimensional vectors into 2D using PCA, t-SNE, or UMAP and visualizes them.

    Args:
        vectors (list or dict): Vectors to project. Can be a list of vectors or a dict 
            mapping labels to vectors.
        labels (list of str, optional): List of labels for the vectors.
        method (str): Method to use for projection ('pca', 'tsne', or 'umap'). 
            Default is 'pca'.
        title (str, optional): Title of the plot.
        color (list of str or str, optional): List of colors for the vectors or a 
            single color.
        figsize (tuple): Figure size as (width, height). Default is (8, 8).
        fontsize (int): Font size for labels. Default is 12.
        perplexity (float, optional): Perplexity parameter for t-SNE. Required if 
            method is 'tsne'.
        filename (str, optional): Path to save the figure.
        adjust_text_labels (bool): Whether to adjust text labels to avoid overlap. 
            Default is False.
        n_neighbors (int): Number of neighbors for UMAP. Default is 15.
        min_dist (float): Minimum distance between points for UMAP. Default is 0.1.
    """
    import matplotlib.pyplot as plt

    # Ensure labels match the number of vectors if provided
    if labels is not None:
        if len(labels) != len(vectors):
            raise ValueError("Number of labels must match number of vectors")

    if isinstance(vectors, dict):
        labels = list(vectors.keys())
        vectors = list(vectors.values())

    vectors = np.array(vectors)

    if method == 'pca':
        from sklearn.decomposition import PCA
        projector = PCA(n_components=2)
        projected_vectors = projector.fit_transform(vectors)
        explained_variance = projector.explained_variance_ratio_
        x_label = f"PC1 ({explained_variance[0]:.2%} variance)"
        y_label = f"PC2 ({explained_variance[1]:.2%} variance)"
    elif method == 'tsne':
        if perplexity is None:
          raise ValueError("Please specify perplexity for T-SNE")
        from sklearn.manifold import TSNE
        projector = TSNE(n_components=2, perplexity=perplexity)
        projected_vectors = projector.fit_transform(vectors)
        x_label = "Dimension 1"
        y_label = "Dimension 2"
    elif method == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("Please install umap-learn package: pip install umap-learn")
        
        projector = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
        projected_vectors = projector.fit_transform(vectors)
        x_label = "UMAP Dimension 1"
        y_label = "UMAP Dimension 2"
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'")

    if isinstance(color, str):
        color = [color] * len(projected_vectors)
    elif isinstance(color, list):
        if len(color) != len(projected_vectors):
            raise ValueError("Number of colors must match number of vectors")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    texts = []
    for i, vector in enumerate(projected_vectors):
        if color:
            ax.scatter(vector[0], vector[1], color=color[i])
        else:
            ax.scatter(vector[0], vector[1])
        if labels:
            text = ax.text(vector[0], vector[1], labels[i], fontsize=fontsize, ha='left')
            texts.append(text)
    if adjust_text_labels and labels:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def get_bias_direction(
    anchors: tuple[np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Compute the direction vector for measuring bias.
    
    Given either a single tuple (pos_anchor, neg_anchor) or a list of tuples,
    computes the direction vector by taking the mean of differences between 
    positive and negative anchor pairs.
    
    Args:
        anchors: A tuple (pos_vector, neg_vector) or list of such tuples.
            Each vector in the pairs should be a numpy array.
    
    Returns:
        numpy.ndarray: The bias direction vector (normalized).
    """
    if isinstance(anchors, tuple):
        anchors = [anchors]
        
    # anchors is now a list of (pos_anchor, neg_anchor) pairs
    diffs = []
    for (pos_vector, neg_vector) in anchors:
        diffs.append(pos_vector - neg_vector)
    
    bias_direction = np.mean(diffs, axis=0)
    # normalize the bias direction
    bias_norm = np.linalg.norm(bias_direction)
    # make sure it's not 0, otherwise make it 1
    if bias_norm == 0:
        bias_norm = 1.0
    return bias_direction / bias_norm

def calculate_bias(
    anchors: tuple[str, str] | list[tuple[str, str]], 
    targets: list[str], 
    word_vectors: Any
) -> np.ndarray:
    """
    Calculate bias scores for target words along an axis defined by anchor pairs.
    
    Args:
        anchors: Tuple or list of tuples defining the bias axis, e.g. ("man", "woman") 
            or [("king", "queen"), ("man", "woman")].
        targets: List of words to calculate bias for.
        word_vectors: Keyed vectors (e.g. from word2vec_model.wv).
    
    Returns:
        numpy.ndarray: Bias scores (dot products) for each target word.
    """
    # Ensure anchors is a list of tuples
    if isinstance(anchors, tuple) and len(anchors) == 2:
        anchors = [anchors]
    if not all(isinstance(pair, tuple) for pair in anchors):
        raise ValueError("anchors must be a tuple or a list of tuples")

    # Get vectors for anchor pairs
    anchor_vectors = [(word_vectors[pos], word_vectors[neg]) for pos, neg in anchors]
    
    # Calculate the bias direction
    bias_direction = get_bias_direction(anchor_vectors)
    
    # Calculate dot products for each target
    target_vectors = [word_vectors[target] for target in targets]
    return np.array([np.dot(vec, bias_direction) for vec in target_vectors])

def project_bias(
    x: tuple[str, str] | list[tuple[str, str]], 
    y: tuple[str, str] | list[tuple[str, str]] | None, 
    targets: list[str], 
    word_vectors: Any,
    title: str | None = None, 
    color: str | list[str] | None = None, 
    figsize: tuple[int, int] = (8, 8),
    fontsize: int = 12, 
    filename: str | None = None, 
    adjust_text_labels: bool = False, 
    disperse_y: bool = False
) -> None:
    """
    Plot words on a 1D or 2D chart by projecting them onto bias axes.
    
    Projects words onto:
      - x-axis: derived from x (single tuple or list of tuples)
      - y-axis: derived from y (single tuple or list of tuples), if provided
    
    Args:
        x: Tuple or list of tuples defining the x-axis bias direction, 
            e.g. ("man", "woman").
        y: Tuple or list of tuples defining the y-axis bias direction, or None 
            for 1D plot.
        targets: List of words to plot.
        word_vectors: Keyed vectors (e.g. from word2vec_model.wv).
        title (str, optional): Title of the plot.
        color: Color(s) for the points. Can be a single color or list of colors.
        figsize (tuple): Figure size as (width, height). Default is (8, 8).
        fontsize (int): Font size for labels. Default is 12.
        filename (str, optional): Path to save the figure.
        adjust_text_labels (bool): Whether to adjust text labels to avoid overlap. 
            Default is False.
        disperse_y (bool): Whether to add random y-dispersion for 1D plots. 
            Default is False.
    """
    import matplotlib.pyplot as plt

    # Input validation
    if isinstance(x, tuple) and len(x) == 2:
        x = [x]
    if not all(isinstance(pair, tuple) for pair in x):
        raise ValueError("x must be a tuple or a list of tuples")

    if y is not None:
        if isinstance(y, tuple) and len(y) == 2:
            y = [y]
        if not all(isinstance(pair, tuple) for pair in y):
            raise ValueError("y must be a tuple, a list of tuples, or None")

    if not isinstance(targets, list):
        raise ValueError("targets must be a list of words to be plotted")

    # Check if all words are in vectors
    missing_targets = [target for target in targets if target not in word_vectors]
    if missing_targets:
        raise ValueError(f"The following targets are missing in vectors and cannot be plotted: {', '.join(missing_targets)}")

    texts = []
    targets = list(set(targets))  # remove duplicates

    # Calculate bias scores
    projections_x = calculate_bias(x, targets, word_vectors)
    projections_y = calculate_bias(y, targets, word_vectors) if y is not None else None

    fig, ax = plt.subplots(figsize=figsize)

    pos_anchors_x = []
    neg_anchors_x = []
    for pair in x:
        pos_anchors_x.append(pair[0])
        neg_anchors_x.append(pair[1]) 
    
    axis_label = f"{', '.join(neg_anchors_x)} {'-'*20} {', '.join(pos_anchors_x)}"
    ax.set_xlabel(axis_label, fontsize=fontsize)

    if projections_y is None:
        # 1D visualization
        if disperse_y:
            y_dispersion = np.random.uniform(-0.1, 0.1, size=projections_x.shape)
            y_dispersion_max = np.max(np.abs(y_dispersion))
        else:
            y_dispersion = np.zeros(projections_x.shape)
            y_dispersion_max = 1

        for i, proj_x in enumerate(projections_x):
            c = color[i] if (isinstance(color, list)) else color
            ax.scatter(proj_x, y_dispersion[i], color=c)
            text = ax.text(proj_x, y_dispersion[i], targets[i],
                           fontsize=fontsize, ha='left')
            texts.append(text)

        # Draw a horizontal axis at y=0
        ax.axhline(0, color='gray', linewidth=0.5)
        # Hide y-ticks
        ax.set_yticks([])
        ax.set_ylim((-y_dispersion_max*1.2, y_dispersion_max*1.2))

    else:
        # 2D visualization
        for i, (proj_x, proj_y) in enumerate(zip(projections_x, projections_y)):
            c = color[i] if (isinstance(color, list)) else color
            ax.scatter(proj_x, proj_y, color=c)
            text = ax.text(proj_x, proj_y, targets[i],
                           fontsize=fontsize, ha='left')
            texts.append(text)

        pos_anchors_y = []
        neg_anchors_y = []
        for pair in y:
            pos_anchors_y.append(pair[0])
            neg_anchors_y.append(pair[1]) 
        
        axis_label_y = f"{', '.join(neg_anchors_y)} {'-'*20} {', '.join(pos_anchors_y)}"
        ax.set_ylabel(axis_label_y, fontsize=fontsize)

    if adjust_text_labels:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()