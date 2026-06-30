import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles=None, nrows=None, ncols=None, figsize=(14,8)):
    """
    Visualization function for displaying multiple images.

    Args:
        images : list of np.ndarray
            List of images (grayscale or RGB).
        titles : list of str, optional
            Titles for each subplot. Must match length of images.
        nrows, ncols : int, optional
            Grid layout. If not provided, inferred from number of images.
        figsize : tuple, optional
            Figure size in inches.
    """
    n = len(images)
    if titles is None:
        titles = ["" for _ in range(n)]
    assert len(titles) == n, "titles length must match images length"

    # Infer grid if not provided
    if nrows is None or ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # flatten for easy iteration

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:  # grayscale
            ax.imshow(img, cmap="gray")
        else:              # RGB
            ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    # Hide unused axes if grid > number of images
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_polygons(polygons, title):
    plt.figure()

    for poly in polygons:
        poly = np.array(poly)
        poly = np.vstack([poly, poly[0]])
        plt.plot(poly[:,0], poly[:,1])

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()