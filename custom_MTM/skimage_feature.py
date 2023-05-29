import numpy as np
from warnings import warn
from scipy import ndimage as ndi
from scipy.spatial import cKDTree, distance

def _get_excluded_border_width(image, min_distance, exclude_border):
    """Return border_width values relative to a min_distance if requested.
    """
    if isinstance(exclude_border, bool):
        border_width = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        border_width = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError("`exclude_border` should have the same length as the "
                             "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError("`exclude_border`, when expressed as a tuple, must only "
                                 "contain ints.")
            if exclude < 0:
                raise ValueError("`exclude_border` can not be a negative value")
        border_width = exclude_border
    else:
        raise TypeError("`exclude_border` must be bool, int, or tuple with the same "
                        "length as the dimensionality of the image.")

    return border_width

def _get_threshold(image, threshold_abs, threshold_rel):
    """Return the threshold value according to an absolute and a relative
    value.
    """
    threshold = threshold_abs if threshold_abs is not None else image.min()
    if threshold_rel is not None:
        threshold = max(threshold, threshold_rel * image.max())

    return threshold

def _get_peak_mask(image, footprint, threshold, mask=None):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint.size == 1 or image.size == 1:
        return image > threshold

    image_max = ndi.maximum_filter(image, footprint=footprint, mode='constant')
    out = image == image_max
    # no peak for a trivial image
    image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
    if image_is_trivial:
        out[:] = False
        if mask is not None:
            # isolated pixels in masked area are returned as peaks
            isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
            out[isolated_px] = True

    out &= image > threshold
    return out

def _exclude_border(label, border_width):
    """Set label border values to 0.
    """
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label

def _ensure_spacing(coord, spacing, p_norm, max_out):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.
    """
    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)
    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    rejected_peaks_indices = set()
    naccepted = 0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            # keep current point and the points at exactly spacing from it
            candidates.remove(idx)
            dist = distance.cdist([coord[idx]], coord[candidates], distance.minkowski, p=p_norm).reshape(-1)
            candidates = [c for c, d in zip(candidates, dist) if d < spacing]
            # candidates.remove(keep)
            rejected_peaks_indices.update(candidates)
            naccepted += 1
            if max_out is not None and naccepted >= max_out:
                break
    # Remove the peaks that are too close to each other
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)
    if max_out is not None:
        output = output[:max_out]

    return output

def ensure_spacing(coords, spacing=1, p_norm=np.inf, min_split_size=50,
                   max_out=None, *, max_split_size=2000):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.
    """
    output = coords
    if len(coords):
        coords = np.atleast_2d(coords)
        if min_split_size is None:
            batch_list = [coords]
        else:
            coord_count = len(coords)
            split_idx = [min_split_size]
            split_size = min_split_size
            while coord_count - split_idx[-1] > max_split_size:
                split_size *= 2
                split_idx.append(split_idx[-1] + min(split_size, max_split_size))
            batch_list = np.array_split(coords, split_idx)

        output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
        for batch in batch_list:
            output = _ensure_spacing(np.vstack([output, batch]), spacing, p_norm, max_out)
            if max_out is not None and len(output) >= max_out:
                break

    return output

def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities)
    coord = np.transpose(coord)[idx_maxsort]

    if np.isfinite(num_peaks):
        max_out = int(num_peaks)
    else:
        max_out = None

    coord = ensure_spacing(coord, spacing=min_distance, p_norm=p_norm, max_out=max_out)
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]

    return coord

def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf, p_norm=np.inf):
    """Find peaks in an image as coordinate list or boolean mask.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    .. versionchanged:: 0.18
        Prior to version 0.18, peaks of the same height within a radius of
        `min_distance` were all returned, but this could cause unexpected
        behaviour. From 0.18 onwards, an arbitrary peak within the region is
        returned. See issue gh-2592.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`.
    threshold_abs : float or None, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates. The coordinates are sorted according to peaks
        values (Larger first). If False, the output will be a boolean
        array shaped as `image.shape` with peaks present at True
        elements. ``indices`` is deprecated and will be removed in
        version 0.20. Default behavior will be to always return peak
        coordinates. You can obtain a mask as shown in the example
        below.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. Internally, a maximum filter is used for finding local
    maxima. This operation dilates the original image. After comparison of the
    dilated and original image, this function returns the coordinates or a mask
    of the peaks where the dilated image equals the original image.

    See also
    --------
    skimage.feature.corner_peaks

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> img2[15, 15, 15] = 1
    >>> peak_idx = peak_local_max(img2, exclude_border=0)
    >>> peak_idx
    array([[10, 10, 10],
           [15, 15, 15]])

    >>> peak_mask = np.zeros_like(img2, dtype=bool)
    >>> peak_mask[tuple(peak_idx.T)] = True
    >>> np.argwhere(peak_mask)
    array([[10, 10, 10],
           [15, 15, 15]])

    """
    if (footprint is None or footprint.size == 1) and min_distance < 1:
        warn("When min_distance < 1, peak_local_max acts as finding "
             "image > max(threshold_abs, threshold_rel * max(image)).",
             RuntimeWarning, stacklevel=2)
    border_width = _get_excluded_border_width(image, min_distance, exclude_border)
    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size, ) * image.ndim, dtype=bool)
    else:
        footprint = np.asarray(footprint)

    if labels is None:
        # Non maximum filter
        mask = _get_peak_mask(image, footprint, threshold)
        mask = _exclude_border(mask, border_width)
        # Select highest intensities (num_peaks)
        coordinates = _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm)
    else:
        _labels = _exclude_border(labels.astype(int, casting="safe"), border_width)
        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks
        labels_peak_coord = []

        for label_idx, roi in enumerate(ndi.find_objects(_labels)):
            if roi is None:
                continue
            # Get roi mask
            label_mask = labels[roi] == label_idx + 1
            # Extract image roi
            img_object = image[roi].copy()
            # Ensure masked values don't affect roi's local peaks
            img_object[np.logical_not(label_mask)] = bg_val
            mask = _get_peak_mask(img_object, footprint, threshold, label_mask)
            coordinates = _get_high_intensity_peaks(img_object, mask, num_peaks_per_label, min_distance, p_norm)
            # transform coordinates in global image indices space
            for idx, s in enumerate(roi):
                coordinates[:, idx] += s.start

            labels_peak_coord.append(coordinates)

        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, 2), dtype=int)

        if len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(image, out, num_peaks, min_distance, p_norm)

    if indices:
        return coordinates
    else:
        out = np.zeros_like(image, dtype=bool)
        out[tuple(coordinates.T)] = True
        return out