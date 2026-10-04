"""Shared RGB [0,1] corruption pipeline for training, evaluation and serving."""
import cv2
import numpy as np

CONDITIONS = ('clean', 'salt_and_pepper', 'gaussian_blur', 'rectangular_occlusion')
SEVERITIES = {
    'salt_and_pepper': [dict(prob=p) for p in (0.03, 0.08, 0.15)],
    'gaussian_blur': [dict(ksize=k, sigma=s) for k, s in ((3, .7), (5, 1.5), (7, 2.5))],
    'rectangular_occlusion': [dict(num_boxes=n, area_ratio=a) for n, a in ((1, .1), (2, .2), (3, .35))],
}


def rectangles(rng, n, area_ratio, size=128):
    """Place non-overlapping rectangles; measured union area within one pixel row."""
    target = round(area_ratio * size * size)
    # Separate horizontal bands guarantee no overlap and bounded execution time.
    bands = np.linspace(0, size, n + 1, dtype=int)
    boxes = []
    for i in range(n):
        area = target // n + (i < target % n)
        height_max = int(bands[i + 1] - bands[i])
        height_min = int(np.ceil(area / size))
        height = int(rng.integers(height_min, height_max + 1))
        width = min(size, max(1, round(area / height)))
        x = int(rng.integers(0, size - width + 1))
        y = int(rng.integers(bands[i], bands[i + 1] - height + 1))
        boxes.append([x, y, x + width, y + height])
    # Pixel rounding must not move a training sample outside the specified range.
    covered = lambda: sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    while covered() < int(np.ceil(.1*size*size)) or covered() > int(np.floor(.35*size*size)):
        grow = covered() < .1*size*size
        for b in boxes:
            width = b[2]-b[0]
            if (grow and width < size) or (not grow and width > 1):
                width += 1 if grow else -1
                b[0] = min(b[0], size-width)
                b[2] = b[0]+width
                break
    return boxes


def specification(rng, condition=None, parameters=None, seed=None):
    condition = condition or CONDITIONS[int(rng.integers(4))]
    p = dict(parameters or {})
    if condition == 'salt_and_pepper':
        p.setdefault('prob', float(rng.uniform(.02, .15)))
    elif condition == 'gaussian_blur':
        p.setdefault('ksize', int(rng.choice([3, 5, 7])))
        p.setdefault('sigma', float(rng.uniform(.5, 2.5)))
    elif condition == 'rectangular_occlusion':
        p.setdefault('num_boxes', int(rng.integers(1, 4)))
        p.setdefault('area_ratio', float(rng.uniform(.1, .35)))
        p.setdefault('boxes', rectangles(rng, p['num_boxes'], p['area_ratio']))
        p['actual_area_ratio'] = sum((b[2]-b[0])*(b[3]-b[1]) for b in p['boxes']) / 128**2
    elif condition != 'clean':
        raise ValueError(f'Unknown condition: {condition}')
    return dict(condition=condition, parameters=p, seed=int(seed if seed is not None else rng.integers(2**31)))


def apply_corruption(image, spec):
    out = image.copy()
    condition, p = spec['condition'], spec['parameters']
    rng = np.random.default_rng(spec['seed'])
    if condition == 'salt_and_pepper':
        mask = rng.random(image.shape[:2]) < p['prob']
        values = rng.integers(0, 2, image.shape[:2]).astype(np.float32)
        out[mask] = values[mask, None]
    elif condition == 'gaussian_blur':
        out = cv2.GaussianBlur(out, (p['ksize'], p['ksize']), p['sigma'])
    elif condition == 'rectangular_occlusion':
        for x1, y1, x2, y2 in p['boxes']:
            out[y1:y2, x1:x2] = 0
    elif condition != 'clean':
        raise ValueError(condition)
    return out.astype(np.float32)
