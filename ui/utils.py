from __future__ import print_function

import inspect
import re
import numpy as np
import cv2
import os
try:
    import pickle as pickle
except ImportError:
    import pickle


def debug_trace():
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def PickleLoad(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    return data


def PickleSave(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def CVShow(im, im_name='', wait=1):
    if len(im.shape) >= 3 and im.shape[2] == 3:
        im_show = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im_show = im

    cv2.imshow(im_name, im_show)
    cv2.waitKey(wait)
    return im_show


def average_image(imgs, weights):
    im_weights = np.tile(weights[:, np.newaxis, np.newaxis, np.newaxis], (1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    imgs_f = imgs.astype(np.float32)
    weights_norm = np.mean(im_weights)
    average_f = np.mean(imgs_f * im_weights, axis=0) / weights_norm
    average = average_f.astype(np.uint8)
    return average


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def grid_vis(X, nh, nw):  # [buggy]
    if X.shape[0] == 1:
        return X[0]

    # nc = 3
    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.shape[-1] == 1:
        X = np.tile(X, [1, 1, 1, 3])

    h, w = X[0].shape[:2]

    if X.dtype == np.uint8:
        img = np.ones((h * nh, w * nw, 3), np.uint8) * 255
    else:
        img = np.ones((h * nh, w * nw, 3), X.dtype)

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    img = np.squeeze(img)
    return img
