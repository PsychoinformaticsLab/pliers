from PIL import ImageDraw
import numpy as np


def _get_coord(d, label, nan=None):
    if not nan:
        nan = {}
    # extract
    coord = [d.get('{0}_{1}'.format(label, ax), None) for ax in ('x', 'y')]
    # replace NaN if possible
    coord = [nan.get(ax, None) if coord[i] is None or np.isnan(coord[i].item()) else coord[i]
             for i, ax in enumerate(('x', 'y'))]
    if any([c is None for c in coord]):
        return None
    return tuple(coord)


def _get_boundingbox(d, img, label):
    poly = (
        _get_coord(d, '{}_vertex1'.format(label),
                   {'x': 0, 'y': 0}),
        _get_coord(d, '{}_vertex2'.format(label),
                   {'x': img.size[0], 'y': 0}),
        _get_coord(d, '{}_vertex3'.format(label),
                   {'x': img.size[1], 'y': img.size[1]}),
        _get_coord(d, '{}_vertex4'.format(label),
                   {'x': 0, 'y': img.size[1]}))
    return poly


def visualize_facial_features(img, features):
    """
    Render a visualization of GoogleVisionAPIFaceExtractor feature

    Parameters
    ----------
    img : Pillow Image instance
      The image to render draw on.
    features : dict-like
      Container with all feature information for the image. This could
      be a row of the data frame produced by the feature extractor.
    """
    f = features
    draw = ImageDraw.Draw(img)
    bb_poly = _get_boundingbox(f, img, 'boundingPoly')
    draw.polygon(bb_poly, outline='yellow')
    fd_poly = _get_boundingbox(f, img, 'fdBoundingPoly')
    draw.polygon(fd_poly, outline='yellow')
    landmarks = [
        (label, _get_coord(f, 'landmark_{}'.format(label))) for label in [
            'CHIN_GNATHION',
            'CHIN_LEFT_GONION',
            'CHIN_RIGHT_GONION',
            'FOREHEAD_GLABELLA',
            'LEFT_EAR_TRAGION',
            'LEFT_EYEBROW_UPPER_MIDPOINT',
            'LEFT_EYE_BOTTOM_BOUNDARY',
            'LEFT_EYE_LEFT_CORNER',
            'LEFT_EYE_PUPIL',
            'LEFT_EYE_RIGHT_CORNER',
            'LEFT_EYE_TOP_BOUNDARY',
            'LEFT_EYE',
            'LEFT_OF_LEFT_EYEBROW',
            'LEFT_OF_RIGHT_EYEBROW',
            'LOWER_LIP',
            'MIDPOINT_BETWEEN_EYES',
            'MOUTH_CENTER',
            'MOUTH_LEFT',
            'MOUTH_RIGHT',
            'NOSE_BOTTOM_CENTER',
            'NOSE_BOTTOM_LEFT',
            'NOSE_BOTTOM_RIGHT',
            'NOSE_TIP',
            'RIGHT_EAR_TRAGION',
            'RIGHT_EYEBROW_UPPER_MIDPOINT',
            'RIGHT_EYE_BOTTOM_BOUNDARY',
            'RIGHT_EYE_LEFT_CORNER',
            'RIGHT_EYE_PUPIL',
            'RIGHT_EYE_RIGHT_CORNER',
            'RIGHT_EYE_TOP_BOUNDARY',
            'RIGHT_EYE',
            'RIGHT_OF_LEFT_EYEBROW',
            'RIGHT_OF_RIGHT_EYEBROW',
            'UPPER_LIP']]
    landmarks = dict([l for l in landmarks if l[1] is not None])
    # try all possible lines
    for p1, p2 in [
            ('CHIN_GNATHION', 'CHIN_LEFT_GONION'),
            ('CHIN_GNATHION', 'CHIN_RIGHT_GONION'),
            ('LEFT_EAR_TRAGION', 'CHIN_LEFT_GONION'),
            ('RIGHT_EAR_TRAGION', 'CHIN_RIGHT_GONION'),
            ('MOUTH_CENTER', 'MOUTH_LEFT'),
            ('MOUTH_CENTER', 'MOUTH_RIGHT'),
            ('UPPER_LIP', 'MOUTH_RIGHT'),
            ('LOWER_LIP', 'MOUTH_RIGHT'),
            ('UPPER_LIP', 'MOUTH_LEFT'),
            ('LOWER_LIP', 'MOUTH_LEFT'),
            ('NOSE_TIP', 'NOSE_BOTTOM_CENTER'),
            ('NOSE_TIP', 'NOSE_BOTTOM_LEFT'),
            ('NOSE_TIP', 'NOSE_BOTTOM_RIGHT'),
            ('NOSE_TIP', 'MIDPOINT_BETWEEN_EYES'),
            ('LEFT_EYE_TOP_BOUNDARY', 'LEFT_EYE_LEFT_CORNER'),
            ('LEFT_EYE_TOP_BOUNDARY', 'LEFT_EYE_RIGHT_CORNER'),
            ('LEFT_EYE_BOTTOM_BOUNDARY', 'LEFT_EYE_LEFT_CORNER'),
            ('LEFT_EYE_BOTTOM_BOUNDARY', 'LEFT_EYE_RIGHT_CORNER'),
            ('RIGHT_EYE_TOP_BOUNDARY', 'RIGHT_EYE_LEFT_CORNER'),
            ('RIGHT_EYE_TOP_BOUNDARY', 'RIGHT_EYE_RIGHT_CORNER'),
            ('RIGHT_EYE_BOTTOM_BOUNDARY', 'RIGHT_EYE_LEFT_CORNER'),
            ('RIGHT_EYE_BOTTOM_BOUNDARY', 'RIGHT_EYE_RIGHT_CORNER'),
            ('LEFT_EYEBROW_UPPER_MIDPOINT', 'LEFT_OF_LEFT_EYEBROW'),
            ('LEFT_EYEBROW_UPPER_MIDPOINT', 'RIGHT_OF_LEFT_EYEBROW'),
            ('RIGHT_EYEBROW_UPPER_MIDPOINT', 'LEFT_OF_RIGHT_EYEBROW'),
            ('RIGHT_EYEBROW_UPPER_MIDPOINT', 'RIGHT_OF_RIGHT_EYEBROW'),
    ]:
        if not (p1 in landmarks and p2 in landmarks):
            continue
        draw.line((landmarks[p1], landmarks[p2]), fill='yellow', width=1)

    for p in [
        'LEFT_EYE',
        'LEFT_EYE_PUPIL',
        'RIGHT_EYE',
        'RIGHT_EYE_PUPIL',
        'FOREHEAD_GLABELLA',
    ]:
        if not p in landmarks:
            continue
        draw.point(landmarks[p], fill='yellow')
