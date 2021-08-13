"""
Defines correction functions for the models' outputs.
"""
from skimage.measure import regionprops, label


def filter_smallest_islets(segmentations, val_class=0, min_area_ratio=0.5):
    """
    Filters a given segmentation by removing segmentation
    the smallest islands in the segmentation.
    -- segmentations: set of output data, of shape (N, h, w)
    -- val_class: segmentation class to be affected: 1 will filter
        the VMax areas, 2 will affect the VCyc areas. 0 will consider
        VMax and VCyc as one class.
    -- min_area_ratio: Minimum area ratio w/ respect to
                       the largest one for a region to be kept
    """
    result = segmentations.copy()

    right_class_seg = segmentations.copy()
    if val_class == 0:
        right_class_seg[segmentations != 0] = 1
    else:
        right_class_seg[segmentations != val_class] = 0
        right_class_seg[segmentations == val_class] = 1

    for i, seg in enumerate(right_class_seg):
        # Computes properties about each connex region
        labeled = label(seg)
        regions = regionprops(labeled)

        # If no segmentation of this class was found
        if not regions:
            continue

        # Retrieves the surface S of the largest regions,
        # and removes any other regions whose surface is
        # inferior to S * min_ratio
        size_threshold = max((r.area for r in regions))
        for i_reg, region in enumerate(regions):
            if region.area < size_threshold * min_area_ratio:
                minr, minc, maxr, maxc = region.bbox
                # Pixels of the region's bbox that are part of the region
                img = region.image
                # Portion of the whole mask that contains the region
                result_area = result[i, minr:maxr, minc:maxc]
                # Erases the region's pixels
                result_area[img] = 0
    return result
