import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm

def show_vectors(vectors, labels=None, draw_arrows=True, draw_axes=True, colors=None, fontsize=12,
                colormap='viridis', figsize=(8, 8), title=None, normalize=False, filename=None, show_components=False):
    """
    Visualizes vectors in 2D using matplotlib.

    Parameters:
    vectors (list of tuples): List of vectors to visualize.
    labels (list of str, optional): List of labels for the vectors. Defaults to None.
    draw_arrows (bool, optional): Whether to draw arrows from the origin to the vectors. Defaults to True.
    draw_axes (bool, optional): Whether to draw x and y axes. Defaults to True.
    colors (list of str, optional): List of colors for the vectors. Defaults to None.
    fontsize (int, optional): Font size for labels. Defaults to 12.
    colormap (str, optional): Colormap to use if colors are not provided. Defaults to 'viridis'.
    title (str, optional): Title of the plot. Defaults to None.
    normalize (bool, optional): Whether to normalize the vectors. Defaults to False.
    filename (str, optional): If provided, saves the figure to the specified file. Defaults to None.
    show_components (bool, optional): Whether to display vector components along axes. Defaults to False.
    """

    vectors = np.array(vectors)
    dim = vectors.shape[1]

    if dim != 2:
        raise ValueError("Only 2D vectors are supported")
    if labels is not None and len(vectors) != len(labels):
        raise ValueError("Number of vectors and labels must match")

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

    if colors is None:
        cmap = cm.get_cmap(colormap)
        colors = [cmap(i / len(vectors)) for i in range(len(vectors))]

    # Calculate plot dimensions and scaling factors for label positioning
    min_x, max_x = min(vectors[:, 0]), max(vectors[:, 0])
    min_y, max_y = min(vectors[:, 1]), max(vectors[:, 1])

    # Calculate the range of the data
    range_x = max(abs(max_x), abs(min_x))
    range_y = max(abs(max_y), abs(min_y))

    # Calculate a relative offset based on the data range (5% of the range)
    x_offset_factor = range_x * 0.05
    y_offset_factor = range_y * 0.05

    fig, ax = plt.subplots(figsize=figsize)

    for i, vector in enumerate(vectors):
        # Draw the main vector arrow
        if draw_arrows:
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=colors[i])
        else:
            ax.scatter(vector[0], vector[1], color=colors[i])

        # Draw vector components if requested
        if show_components:
            # Only draw y-projection if x component is non-zero
            if vector[0] != 0:
                # Vertical projection line
                ax.plot([vector[0], vector[0]], [0, vector[1]], color=colors[i], linestyle=':', alpha=0.5)

                # X-component label with scaled offset
                x_label_y_pos = -y_offset_factor if vector[1] > 0 else y_offset_factor
                x_label_va = 'top' if vector[1] > 0 else 'bottom'
                ax.text(vector[0], x_label_y_pos, f"{vector[0]:.2f}", color=colors[i],
                       ha='center', va=x_label_va, fontsize=fontsize-2)

            # Only draw x-projection if y component is non-zero
            if vector[1] != 0:
                # Horizontal projection line
                ax.plot([0, vector[0]], [vector[1], vector[1]], color=colors[i], linestyle=':', alpha=0.5)

                # Y-component label with scaled offset
                y_label_x_pos = -x_offset_factor if vector[0] > 0 else x_offset_factor
                y_label_ha = 'right' if vector[0] > 0 else 'left'
                ax.text(y_label_x_pos, vector[1], f"{vector[1]:.2f}", color=colors[i],
                       ha=y_label_ha, va='center', fontsize=fontsize-2)

        # Add label if provided
        if labels:
            ax.text(vector[0], vector[1], labels[i], fontsize=fontsize, ha='left')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    if draw_axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    
    ax.set_xlim(-range_x - x_offset_factor * 10, range_x + x_offset_factor * 10)
    ax.set_ylim(-range_y - y_offset_factor * 10, range_y + y_offset_factor * 10)
    plt.tight_layout()

    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()