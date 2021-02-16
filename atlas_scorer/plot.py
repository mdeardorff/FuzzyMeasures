from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from atlas_scorer.score import ConfusionMatrixResult


def confusion_matrix_plot(c_matrix: ConfusionMatrixResult,
                          color_lims=None,
                          ax=None):
    """
    Visualize a ConfusionMatrixResult as a table

    Args:
        c_matrix (ConfusionMatrixResult): The confusion matrix
        color_lims (List[float] or None): A list of length 2, containing the
            min and max values for clipping the colors in the displayed table.
            If color_lims is None, the default values are [0, 1] if the
            c_matrix.normalize is True, and [0, count_matrix.max()] otherwise.
        ax (mpl.axes.Axes, or None): Axes instance (im_ax.axes).  Default is
            None.

    Returns:
        (ConfusionMatrixResult): The input ConfusionMatrixResult with the
            following attributes set:
                im_ax (mpl.image.AxesImage): AxesImage instance
                ax (mpl.axes.Axes): Axes instance (im_ax.axes)
                fig (mpl.figure.Figure): Figure instance (im_ax.figure)
                text_annots (List[mpl.text.Text]): Text annotations for each
                    cell

    """
    n_decls = c_matrix.num_decls
    n_annots = c_matrix.num_annots
    u_decl_classes = c_matrix.u_decl_classes
    u_annot_classes = c_matrix.u_annot_classes
    count_matrix = c_matrix.count_matrix

    if color_lims is None:
        color_lims = [0, 1] if c_matrix.normalize else [0, count_matrix.max()]

    color_thresh = np.mean(color_lims)

    decl_labels = [f'{u}[{n_decls[i][0]:.0f}]' for i, u in
                   enumerate(u_decl_classes)]
    annot_labels = [f'{u}[{n_annots[i][0]:.0f}]' for i, u in
                    enumerate(u_annot_classes)]

    # Plot the confusion matrix
    im, _ = heatmap(
        count_matrix, row_labels=annot_labels, col_labels=decl_labels,
        xlabel='Declaration', ylabel='Truth', xticks_rotation=45,
        display_cbar=False, clims=color_lims, ax=ax)

    # Annotate the confusion matrix
    text_annots = annotate_heatmap(im, threshold=color_thresh)

    # Save out references to matplotlib handles for different plot elements
    c_matrix.im_ax = im
    c_matrix.ax = im.axes
    c_matrix.fig = im.figure
    c_matrix.text_annots = text_annots

    return c_matrix


# Adapted from
# https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, cmap='binary', clims=None,
            cbar_kw={}, cbarlabel="", display_cbar=True, xlabel="", ylabel="",
            title="", xticks_rotation='horizontal', hide_spine=False,
            grid_color="k", grid_lw=2, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Args:
        data (np.ndarray): A 2D numpy array of shape (N, M).
        row_labels (list[str]): A list or array of length N with the labels for
            the rows. (Y-axis)
        col_labels (list[str]): A list or array of length M with the labels for
            the columns. (X-axis)
        ax (mpl.axes.Axes, optional): A `matplotlib.axes.Axes` instance to which
            the heatmap is plotted.  If not provided, use current axes or create
            a new one.  Optional.
        cmap (str, or mpl.colors.Colormap): str or matplotlib Colormap
            recognized by matplotlib; default='binary'
        clims (Iterable, optional): 2-element iterable mapping to min and max
            values in the colormap; Default of **None** results in min and max
            of ``data`` determining the color limits
        cbar_kw (dict, optional): A dictionary with arguments to
            `matplotlib.Figure.colorbar`.
        cbarlabel (str, optional): The label for the colorbar.
        display_cbar (bool, optional): Whether to display colorbar;
            Default **True**
        xlabel (str, optional): Label for x-axis; Default is ""
        ylabel (str, optional): Label for y-axis; Default is ""
        title (str, optional): Label for plot title; Default is ""
        xticks_rotation (str or float, optional): {'vertical', 'horizontal'} or
            float; default='horizontal'
        hide_spine (bool, optional): Whether to hide spine (outline for plotted
            image); Default is **False**
        grid_color: Any matplotlib compatible color specification for the grid
            drawn between each cell; Default is black
        grid_lw (float, optional): Linewidth for the grid; Default is **2**.
            Note: Value of 1 can look wrong due to 1px alignment issues between
            the grid and the underlying imshow plot
        **kwargs: All other arguments are forwarded to `imshow`.

    Returns:
        im (mpl.image.AxesImage):
        cbar (mpl.colorbar.Colorbar): None if colorbar was disabled

    Example:
        >>> labels = ['civilian', 'military', 'N/A[NaN]']
        >>> data = np.array([[148, 43, 42],[187, 278, 110],[9, 450, 0]])
        >>> # Generate heatmap plot
        >>> im, _ = heatmap(data, row_labels=labels, col_labels=labels,
        >>>     xlabel='Declaration', ylabel='Truth', xticks_rotation=45,
        >>>     display_cbar=False)
        >>> # Annotate heatmap with appropriate font-size
        >>> tex = annotate_heatmap(im)
    """
    assert data.ndim == 2, 'Only 2D ndarray is supported by plot_matrix'
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if clims is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = clims

    # Plot the heatmap
    im = ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=vmin,
                   vmax=vmax, **kwargs)

    # Create colorbar
    cbar = None
    if display_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim((data.shape[0] - 0.5, -0.5))
    ax.set_title(title)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    if xticks_rotation == 'horizontal' or xticks_rotation == 0:
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation, ha="center",
                 rotation_mode="anchor")
    else:
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation, ha="right",
                 rotation_mode="anchor")

    # Turn spines off if `hide_spine` == True.
    for edge, spine in ax.spines.items():
        spine.set_visible(not hide_spine)
        spine.set_linewidth(grid_lw)

    # Color grid
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color=grid_color, linestyle='-', linewidth=grid_lw)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    return im, cbar


def annotate_heatmap(im, data=None, valfmt=None, float_precision=2,
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap with values for each pixel. Font-size is
    auto-calculated by default and resize function for the window automatically
    handles rescaling of the font-size

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt (dict)
        The format of the annotations inside the heatmap.  A dict with of format:
        {'int': ..., 'float': ...} where values should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    float_precision (int)
        Overrides default float formatter and only applicable if `valfmt` is
        not provided.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        # returns np.ma (MaskedArray). Set masked values to nan and get
        # underlying data
        data = im.get_array()
        data.set_fill_value(np.nan)
        data = data.data

    int_formatter = mpl.ticker.StrMethodFormatter("{x:0.0f}")
    float_formatter = mpl.ticker.StrMethodFormatter(f"{{x:.{float_precision}f}}")
    DEFAULT_FORMATTER = {'int': int_formatter, 'float': float_formatter}

    # Modify default formatting for integer vs float
    if valfmt is not None:
        # Wrap provided formatter if aprovided as a string specification
        for k, v in valfmt.items():
            if isinstance(v, str):
                valfmt[k] = mpl.ticker.StrMethodFormatter(v)
    else:
        valfmt = {}

    valfmt = {**DEFAULT_FORMATTER, **valfmt}  # Merge formatters with defaults

    # Calculate font-size for annotation and merge with supplied kwargs
    fs = calc_best_fontsize(im)
    text_kwargs = {'fontsize': fs}
    text_kwargs.update(textkw)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(text_kwargs)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            c_data = data[i, j]
            # Check if approximately int or not and format accordingly
            if np.isclose(np.mod(c_data, 1), 0, atol=1e-5):
                text = im.axes.text(j, i, valfmt['int'](c_data, None), **kw)
            else:
                text = im.axes.text(j, i, valfmt['float'](c_data, None), **kw)
            texts.append(text)

    def on_resize(event):
        fs = calc_best_fontsize(im)
        for t in texts:
            t.set_fontsize(fs)

    # Bind callback to resize font when window is resized
    cid = im.axes.figure.canvas.mpl_connect('resize_event', on_resize)
    return texts


def calc_best_fontsize(im):
    data = im.get_array()
    nrows, ncols = data.shape
    ax = im.axes
    fig = ax.figure
    w_px, h_px = fig.canvas.get_width_height()
    im_xyxy = ax.get_position().extents  # Normalized
    im_w_px = (im_xyxy[2] - im_xyxy[0]) * w_px
    im_h_px = (im_xyxy[3] - im_xyxy[1]) * h_px

    ratio_num = np.array([im_w_px, im_h_px])

    magic_number = 80
    if (ncols < magic_number) and (nrows < magic_number):
        ratio = min(ratio_num / np.array([nrows, ncols]))
    elif ncols < magic_number:
        ratio = ratio_num[1] / ncols
    elif nrows < magic_number:
        ratio = ratio_num[0] / nrows
    else:
        ratio = 1
    # fs = min(maxFontSize, ceil(ratio / 4)); % the gold formula
    fs = ceil(ratio / 4)
    if fs < 4:  # Font sizes less than 4 still look like crap
        fs = 0
    return fs
