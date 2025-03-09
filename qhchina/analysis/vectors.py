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

def get_bias_direction(anchors):
    """
    Given either a single tuple (pos_anchor, neg_anchor) or a list of tuples,
    compute the direction vector for measuring bias by taking the mean of 
    differences between positive and negative anchor pairs.
    
    Parameters:
    anchors: A tuple (pos_vector, neg_vector) or list of such tuples
            Each vector in the pairs should be a numpy array
    
    Returns:
    numpy array representing the bias direction vector (unnormalized)
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
    return bias_direction / bias_norm

def calculate_bias(anchors, targets, word_vectors):
    """
    Calculate bias scores for target words along an axis defined by anchor pairs.
    
    Parameters:
    anchors: tuple or list of tuples, e.g. ("man", "woman") or [("king", "queen"), ("man", "woman")]
    targets: list of words to calculate bias for
    word_vectors: keyed vectors (e.g. from word2vec_model.wv)
    
    Returns:
    numpy array of bias scores (dot products) for each target word
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

def project_bias(x, y, targets, word_vectors,
                    title=None, color=None, figsize=(8,8),
                    fontsize=12, filename=None, adjust_text_labels=False, disperse_y=False):
    """
    Plots words on either a 1D or 2D chart by projecting them onto:
      - axis_x: derived from x (single tuple or list of tuples)
      - axis_y: derived from y (single tuple or list of tuples), if provided

    Parameters remain the same as before, but calculation of bias scores is now handled separately.
    """
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
        # 2D visualization
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