import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from adjustText import adjust_text

def project_2d(vectors, 
               labels=None, 
               method='pca', 
               title=None, 
               color=None, 
               figsize=(8,8), 
               fontsize=12, 
               perplexity=None,
               filename=None,
               adjust_text_labels=False):
    """
    Projects high-dimensional vectors into 2D using PCA or t-SNE and visualizes them.

    Parameters:
    vectors (list of lists or dict {label: vector}): Vectors to project.
    labels (list of str, optional): List of labels for the vectors. Defaults to None.
    method (str, optional): Method to use for projection ('pca' or 'tsne'). Defaults to 'pca'.
    title (str, optional): Title of the plot. Defaults to None.
    color (list of str or str, optional): List of colors for the vectors or a single color. Defaults to None.
    """
    # Ensure vectors are lists or tuples
    if not isinstance(vectors, (list, tuple, dict)):
        raise ValueError("vectors must be a list, tuple, or dict")

    # If vectors is a list or tuple, ensure each element is a list or tuple
    if isinstance(vectors, (list, tuple)):
        if not all(isinstance(vec, (list, tuple)) for vec in vectors):
            raise ValueError("Each vector must be a list or tuple")

    # Ensure labels match the number of vectors if provided
    if labels is not None:
        if len(labels) != len(vectors):
            raise ValueError("Number of labels must match number of vectors")

    if isinstance(vectors, dict):
        labels = list(vectors.keys())
        vectors = list(vectors.values())

    vectors = np.array(vectors)

    if method == 'pca':
        projector = PCA(n_components=2)
        projected_vectors = projector.fit_transform(vectors)
        explained_variance = projector.explained_variance_ratio_
        x_label = f"PC1 ({explained_variance[0]:.2%} variance)"
        y_label = f"PC2 ({explained_variance[1]:.2%} variance)"
    elif method == 'tsne':
        if perplexity is None:
          raise ValueError("Please specify perplexity for T-SNE")
        projector = TSNE(n_components=2, perplexity=perplexity)
        projected_vectors = projector.fit_transform(vectors)
        x_label = "Dimension 1"
        y_label = "Dimension 2"
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

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
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def get_axis(anchors):
    """
    Given either a single tuple (pos_anchor, neg_anchor) or a list of tuples,
    compute the mean difference vector (pos - neg) and return the normalized axis.
    Assumes anchors are already vectors.
    """
    if isinstance(anchors, tuple):
        anchors = [anchors]
        
    # anchors is now a list of (pos_anchor, neg_anchor) pairs
    diffs = []
    for (pos_vector, neg_vector) in anchors:
        diffs.append(pos_vector - neg_vector)
    
    axis = np.mean(diffs, axis=0)  # average across all pairs
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        # Edge case: if all diffs sum to zero (very unlikely), return zeros
        return axis
    return axis / axis_norm

def project_bias(x, y, targets, word_vectors,
                    title=None, color=None, figsize=(8,8),
                    fontsize=12, filename=None, adjust_text_labels=False, disperse_y=False):
    """
    Plots words on either a 1D or 2D chart by projecting them onto:
      - axis_x: derived from x (single tuple or list of tuples)
      - axis_y: derived from y (single tuple or list of tuples), if provided

    :param x: tuple or a list of tuples for the horizontal axis, e.g. ("同志","敌人") or [("man","woman"), ...]
    :param y: tuple or a list of tuples for the vertical axis, or None for 1D
    :param targets: list of words to plot
    :param word_vectors: keyed vectors (e.g. from word2vec_model.wv)
    :param title: optional plot title
    :param color: either a single color or list of colors for the words
    :param figsize: (width, height) in inches
    :param fontsize: font size for word labels
    :param filename: if given, saves the figure to disk
    :param adjust_text_labels: if True, tries to automatically adjust text to reduce overlap
    """
    # Ensure x is a list of tuples
    if isinstance(x, (tuple, list)) and len(x) == 2:
        x = [x]
    if not all(isinstance(pair, tuple) for pair in x):
        raise ValueError("x must be a tuple or a list of tuples")

    # Ensure y is a list of tuples or None
    if y is not None:
        if isinstance(y, (tuple, list)) and len(y) == 2:
            y = [y]
        if not all(isinstance(pair, tuple) for pair in y):
            raise ValueError("y must be a tuple, a list of tuples, or None")

    # Ensure targets is a list
    if not isinstance(targets, list):
        raise ValueError("targets must be a list of words to be plotted")

    # Check if all words are in vectors
    missing_targets = [target for target in targets if target not in word_vectors]
    if missing_targets:
        raise ValueError(f"The following targets are missing in vectors and cannot be plotted: {', '.join(missing_targets)}")

    texts = []

    axis_x_unit = get_axis([(word_vectors[pos_target], word_vectors[neg_target]) 
                            for pos_target, neg_target in x])

    axis_y_unit = None
    if y is not None:
        axis_y_unit = get_axis([(word_vectors[pos_target], word_vectors[neg_target]) 
                                for pos_target, neg_target in y])

    targets = list(set(targets))  # remove duplicates
    target_vectors = [word_vectors[target] for target in targets]

    projections_x = np.array([
        np.dot(vec, axis_x_unit) for vec in target_vectors
    ])

    if axis_y_unit is not None:
        projections_y = np.array([
            np.dot(vec, axis_y_unit) for vec in target_vectors
        ])

    fig, ax = plt.subplots(figsize=figsize)

    if axis_y_unit is None:
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

        pos_anchors = []
        neg_anchors = []
        for pair in x:
            pos_anchors.append(pair[0])
            neg_anchors.append(pair[1]) 
        
        axis_label = f"{', '.join(neg_anchors)} {'-'*20} {', '.join(pos_anchors)}"
        ax.set_xlabel(axis_label, fontsize=fontsize)

        # Hide y-ticks
        ax.set_yticks([])
        ax.set_ylim((-y_dispersion_max*1.2, y_dispersion_max*1.2))

    else:
        # 2D case: we have both x and y
        for i, (proj_x, proj_y) in enumerate(zip(projections_x, projections_y)):
            c = color[i] if (isinstance(color, list)) else color
            ax.scatter(proj_x, proj_y, color=c)
            text = ax.text(proj_x, proj_y, targets[i],
                           fontsize=fontsize, ha='left')
            texts.append(text)

        pos_anchors_x = []
        neg_anchors_x = []
        for pair in x:
            pos_anchors_x.append(pair[0])
            neg_anchors_x.append(pair[1]) 
        
        axis_label_x = f"{', '.join(neg_anchors_x)} {'-'*20} {', '.join(pos_anchors_x)}"
        ax.set_xlabel(axis_label_x, fontsize=fontsize)

        pos_anchors_y = []
        neg_anchors_y = []
        for pair in y:
            pos_anchors_y.append(pair[0])
            neg_anchors_y.append(pair[1]) 
        
        axis_label_y = f"{', '.join(neg_anchors_y)} {'-'*20} {', '.join(pos_anchors_y)}"
        ax.set_ylabel(axis_label_y, fontsize=fontsize)

    if adjust_text_labels:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))