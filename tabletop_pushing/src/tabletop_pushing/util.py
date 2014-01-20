from math import pi, sin, cos
import numpy as np

def sign(x):
    if x < 0:
        return -1
    return 1

def subPIAngle(theta):
    while theta < -pi:
        theta += 2.0*pi
    while theta > pi:
        theta -= 2.0*pi
    return theta

def trigAugState(X, ndx, remove_old=False):
    X_aug = []
    if remove_old:
        for i, x in enumerate(X):
            if i in ndx:
                continue
            else:
                X_aug.append(x)
    else:
        X_aug = X[:]
    for i in ndx:
        X_aug = np.append(X_aug, [sin(X[i]), cos(X[i])])
    return np.asarray(X_aug)
