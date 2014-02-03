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


def subPIDiff(a, b):
    return subPIAngle(subPIAngle(a) - subPIAngle(b))

def subPIDiffNP(a, b):
    diff = np.zeros(len(a))
    for i, (a_i, b_i) in enumerate(zip(a,b)):
        diff[i] = abs(subPIAngle(subPIAngle(a_i) - subPIAngle(b_i)))
    return diff

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
