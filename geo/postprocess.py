import numpy as np
import cv2
from skimage.morphology import skeletonize
from shapely.geometry import LineString
import geojson


def mask_to_geojson(mask, out_path, threshold=0.3, min_length=30):
    """
    Convert road probability mask to centerline GeoJSON
    """

    # Binary mask
    binary = (mask > threshold).astype(np.uint8)

    # Skeletonize -> 1 pixel wide centerlines
    skel = skeletonize(binary).astype(np.uint8)

    # Find contours on skeleton
    contours, _ = cv2.findContours(
        skel,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    features = []

    for cnt in contours:
        if len(cnt) < min_length:
            continue

        coords = [(int(p[0][0]), int(p[0][1])) for p in cnt]

        try:
            line = LineString(coords)
            features.append(geojson.Feature(geometry=line))
        except:
            continue

    fc = geojson.FeatureCollection(features)

    with open(out_path, "w") as f:
        geojson.dump(fc, f)

    return out_path
