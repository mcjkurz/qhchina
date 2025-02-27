import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
import math

def show_vectors(vectors, labels=None, draw_arrows=True, draw_axes=True, colors=None, colormap='viridis', title=None, normalize=False, filename=None):
    """
    Visualizes vectors in 2D using matplotlib.

    Parameters:
    vectors (list of lists or np.ndarray): List of vectors to visualize.
    labels (list of str, optional): List of labels for the vectors. Defaults to None.
    draw_arrows (bool, optional): Whether to draw arrows from the origin to the vectors. Defaults to True.
    colors (list of str, optional): List of colors for the vectors. Defaults to None.
    colormap (str, optional): Colormap to use if colors are not provided. Defaults to 'viridis'.
    title (str, optional): Title of the plot. Defaults to None.
    normalize (bool, optional): Whether to normalize the vectors. Defaults to False.
    """
    if isinstance(vectors, dict):
        vectors = list(vectors.values())
        labels = list(vectors.keys())

    vectors = np.array(vectors)
    dim = vectors.shape[1]

    if dim not in [2]:
        raise ValueError("Only 2D vectors are supported")

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

    if colors is None:
        cmap = cm.get_cmap(colormap)
        colors = [cmap(i / len(vectors)) for i in range(len(vectors))]

    fig, ax = plt.subplots()

    for i, vector in enumerate(vectors):
        if draw_arrows:
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=colors[i])
        else:
            ax.scatter(vector[0], vector[1], color=colors[i])
        if labels:
            ax.text(vector[0], vector[1], labels[i], fontsize=12, ha='left')

    # Extract min and max values
    min_x, max_x = min(vectors[:, 0]), max(vectors[:, 0])
    min_y, max_y = min(vectors[:, 1]), max(vectors[:, 1])

    range_x = max(abs(max_x), abs(min_x))
    range_y = max(abs(max_y), abs(min_y))

    tick_size_x = pow(10, math.floor(math.log10(range_x))) if range_x > 1 else 1
    tick_size_y = pow(10, math.floor(math.log10(range_y))) if range_y > 1 else 1
    
    x_lim = (-tick_size_x if min_x >= 0 else min_x - tick_size_x, max_x + tick_size_x if max_x > 0 else tick_size_x)
    y_lim = (-tick_size_y if min_y >= 0 else min_y - tick_size_y, max_y + tick_size_y if max_y > 0 else tick_size_y)

    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    
    if tick_size_x == tick_size_y:
        ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if draw_axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()