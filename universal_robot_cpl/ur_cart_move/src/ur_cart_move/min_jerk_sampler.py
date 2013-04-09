#! /usr/bin/python

import numpy as np

def min_jerk_gen(x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d):
    #print 'min_jerk_gen'
    #print x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d
    A = np.mat([[  d**3,     d**4,     d**5],
                [3*d**2,   4*d**3,   5*d**4],
                [  6*d,  12*d**2,  20*d**3]])

    b = np.mat([[x_f-(x_i+ xd_i*d+xdd_i*d**2)],
                [xd_f-(xd_i+2*xdd_i*d)],
                [xdd_f-(2*xdd_i)]])

    a = np.zeros(6)
    a[0:3] = [x_i, xd_i, xdd_i]
    a[3:6] = np.linalg.solve(A,b).T.A[0]
    ad = np.array([a[1], 2*a[2], 3*a[3], 4*a[4], 5*a[5]])
    add = np.array([2*a[2], 6*a[3], 12*a[4], 20*a[5]])
    return a[::-1], ad[::-1], add[::-1]

def min_jerk_sample(a, d, max_dx, min_dt):
    a = np.array(a)
    xk = [np.polyval(a,0.)]
    tk = [0.]
    k = 0
    while True:
        good_knots = [d, tk[k]+min_dt]
        for msign in [-1., 1.]:
            aptest = a.copy()
            aptest[-1] -= np.polyval(a,tk[k]) + msign * max_dx
            for r in np.roots(aptest):
                if np.iscomplexobj(r): 
                    if np.imag(r) == 0:
                        r = np.real(r)
                    else:
                        continue
                r = float(r)
                if r > tk[k]:
                    good_knots.append(r)
        k += 1
        tk.append(min(good_knots))
        xk.append(np.polyval(a, tk[-1]))
        if tk[-1] == d:
            tk.remove(tk[-2])
            xk.remove(xk[-2])
            return tk, xk

def sample_min_jerk_knots(x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d, max_dx, min_dt):
    a, ad, add = min_jerk_gen(x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d)
    return min_jerk_sample(a, d, max_dx, min_dt)

def main():
    x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d, max_dx, min_dt = (
     0.,   0.,    0.,  1., -400.,   0.,1.,    10.,    0.1)
    tk, xk = sample_min_jerk_knots(x_i, xd_i, xdd_i, x_f, xd_f, xdd_f, d, max_dx, min_dt)
    print tk, xk

if __name__ == "__main__":
    main()
