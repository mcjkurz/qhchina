import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    vectors (list of lists or np.ndarray or dict): List of vectors to project.
    labels (list of str, optional): List of labels for the vectors. Defaults to None.
    method (str, optional): Method to use for projection ('pca' or 'tsne'). Defaults to 'pca'.
    title (str, optional): Title of the plot. Defaults to None.
    color (list of str or str, optional): List of colors for the vectors or a single color. Defaults to None.
    """
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
        try:
            from adjustText import adjust_text
        except ImportError:
            print("adjustText not available, please install via pip.")
            adjust_text = None
        if adjust_text:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

def get_axis(anchors, vectors):
    """
    Given either a single tuple (pos_anchor, neg_anchor) or a list of tuples,
    compute the mean difference vector (pos - neg) and return the normalized axis.
    """
    if isinstance(anchors, tuple):
        anchors = [anchors]
        
    # anchors is now a list of (pos_anchor, neg_anchor) pairs
    diffs = []
    for (pos_word, neg_word) in anchors:
        diffs.append(vectors[pos_word] - vectors[neg_word])
    
    axis = np.mean(diffs, axis=0)  # average across all pairs
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        # Edge case: if all diffs sum to zero (very unlikely), return zeros
        return axis
    return axis / axis_norm

def project_bias(x, y, words, vectors,
                    title=None, color=None, figsize=(8,8),
                    fontsize=12, filename=None, adjust_text_labels=False, disperse_y=False):
    """
    Plots words on either a 1D or 2D chart by projecting them onto:
      - axis_x: derived from x (single tuple or list of tuples)
      - axis_y: derived from y (single tuple or list of tuples), if provided

    :param x: list of tuples for the horizontal axis, e.g. ("同志","敌人") or [("man","woman"), ...]
    :param y: list of tuples for the vertical axis, or None for 1D
    :param words: list of words to plot
    :param vectors: keyed vectors (e.g. from word2vec_model.wv)
    :param title: optional plot title
    :param color: either a single color or list of colors for the words
    :param figsize: (width, height) in inches
    :param fontsize: font size for word labels
    :param filename: if given, saves the figure to disk
    :param adjust_text_labels: if True, tries to automatically adjust text to reduce overlap
    """
    texts = []

    axis_x_unit = get_axis(x, vectors)

    axis_y_unit = None
    if y is not None:
        axis_y_unit = get_axis(y, vectors)

    words = list(set(words))  # remove duplicates
    target_vectors = [vectors[word] for word in words]

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
            text = ax.text(proj_x, y_dispersion[i], words[i],
                           fontsize=fontsize, ha='left')
            texts.append(text)

        # Draw a horizontal axis at y=0
        ax.axhline(0, color='gray', linewidth=0.5)

        pos_words = []
        neg_words = []
        for pair in x:
            pos_words.append(pair[0])
            neg_words.append(pair[1]) 
        
        axis_label = f"{', '.join(neg_words)} {'-'*20} {', '.join(pos_words)}"
        ax.set_xlabel(axis_label, fontsize=fontsize)

        # Hide y-ticks
        ax.set_yticks([])
        #ax.set_xlim((min_projection_x*1.05, max_projection_x*1.05))
        ax.set_ylim((-y_dispersion_max*1.2, y_dispersion_max*1.2))

    else:
        # 2D case: we have both x and y
        for i, (proj_x, proj_y) in enumerate(zip(projections_x, projections_y)):
            c = color[i] if (isinstance(color, list)) else color
            ax.scatter(proj_x, proj_y, color=c)
            text = ax.text(proj_x, proj_y, words[i],
                           fontsize=fontsize, ha='left')
            texts.append(text)

        pos_words_x = []
        neg_words_x = []
        for pair in x:
            pos_words_x.append(pair[0])
            neg_words_x.append(pair[1]) 
        
        axis_label_x = f"{', '.join(neg_words_x)} {'-'*20} {', '.join(pos_words_x)}"
        ax.set_xlabel(axis_label_x, fontsize=fontsize)

        pos_words_y = []
        neg_words_y = []
        for pair in y:
            pos_words_y.append(pair[0])
            neg_words_y.append(pair[1]) 
        
        axis_label_y = f"{', '.join(neg_words_y)} {'-'*20} {', '.join(pos_words_y)}"
        ax.set_ylabel(axis_label_y, fontsize=fontsize)

    if adjust_text_labels:
        try:
            from adjustText import adjust_text
        except ImportError:
            print("adjustText not available, please install via pip.")
            adjust_text = None
        if adjust_text:
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