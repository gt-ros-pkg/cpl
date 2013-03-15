#!/usr/bin/python

import numpy as np
from collections import defaultdict

import roslib
roslib.load_manifest('ur_cart_move')

import rospy

from ur_cart_move.ur_cart_move import load_ur_robot
from hrl_geom.pose_converter import PoseConv

def main():
    np.set_printoptions(precision=4, suppress=True)
    rospy.init_node("ur_cart_move")
    arm, kin, arm_behav = load_ur_robot()

    if False:
        # I think we can conclude that using an initial guess of the direction doesn't help much
        def test_jac_add(delta_mult):
            q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
            x_rand = kin.forward(q_rand)
            x_twist = (np.random.normal(0,0.03,3).tolist(), [0.]*3)
            q_delta, resid = kin.inverse_velocity(x_twist, q_rand)
            if resid != 0.0:
                q_delta *= 0.
            x_solve = x_rand.copy()
            x_solve[:3,3] += np.mat(x_twist[0]).T
            return kin.inverse(x_solve, q_rand + q_delta/np.linalg.norm(q_delta)*delta_mult)
        n = 1000
        num_fail = 0
        for i in range(n):
            if test_jac_add(0.3) is None:
                num_fail += 1
        print num_fail
        num_fail = 0
        for i in range(n):
            if test_jac_add(0.) is None:
                num_fail += 1
        print num_fail
        
    if False:
        # measures the effect of how many restarts are required to get a solution based on
        # increasing changes in q
        # [[ 0.9975  0.0015  0.      0.    ]
        #  [ 0.9735  0.016   0.0065  0.0025]
        #  [ 0.941   0.042   0.01    0.0055]
        #  [ 0.9005  0.0675  0.0185  0.003 ]
        #  [ 0.8285  0.1035  0.0405  0.0135]
        #  [ 0.779   0.133   0.05    0.0185]
        #  [ 0.6365  0.2065  0.083   0.0355]
        #  [ 0.596   0.2125  0.0865  0.0415]]
        #[0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 4.0]
        def kin_random_restart(x, q_guess, sigma_add):
            num_restart = 0
            while not rospy.is_shutdown():
                q_try = q_guess + np.random.normal(0, sigma_add, 6)
                if kin.inverse(x, q_try) is not None:
                    return num_restart
                if num_restart == 50:
                    return num_restart
                num_restart += 1
            return -1
        sigmas = [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 3.0, 4.0]
        prcs = np.zeros((len(sigmas), 4))
        for j, sigma in enumerate(sigmas):
            rr_list = []
            n = 2000
            for i in range(n):
                q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                x_rand = kin.forward(q_rand)
                rr_list.append(kin_random_restart(x_rand, q_rand, sigma))
            d = defaultdict(int)
            for rr in rr_list:
                d[rr] += 1
            print d
            for i in range(4):
                prcs[j][i] = float(d[i]) / n
        print prcs
        print sigmas

    if False:
        # measures the effect of how many restarts are required to get a solution based on
        # increasing changes in x
        # 
        # defaultdict(<type 'int'>, {0: 1482, 4: 1, 7: 1, 11: 2, 12: 1, 13: 1, 14: 1, 15: 511})
        # defaultdict(<type 'int'>, {0: 1478, 1: 19, 2: 11, 3: 7, 4: 5, 5: 4, 6: 4, 7: 2, 8: 1, 10: 1, 11: 2, 15: 466})
        # defaultdict(<type 'int'>, {0: 1448, 1: 65, 2: 24, 3: 7, 4: 6, 5: 2, 6: 3, 7: 2, 8: 3, 9: 2, 10: 3, 12: 2, 13: 3, 14: 2, 15: 428})
        # defaultdict(<type 'int'>, {0: 1403, 1: 103, 2: 27, 3: 15, 4: 9, 5: 6, 6: 5, 7: 4, 8: 2, 9: 4, 10: 2, 11: 2, 12: 5, 13: 2, 15: 411})
        # defaultdict(<type 'int'>, {0: 1304, 1: 168, 2: 54, 3: 27, 4: 16, 5: 14, 6: 9, 7: 4, 8: 4, 9: 2, 10: 3, 11: 3, 12: 1, 13: 3, 14: 6, 15: 382})
        # defaultdict(<type 'int'>, {0: 1223, 1: 232, 2: 83, 3: 31, 4: 25, 5: 11, 6: 7, 7: 7, 8: 7, 9: 5, 10: 5, 11: 4, 12: 1, 13: 2, 14: 2, 15: 355})
        # [[ 0.741   0.      0.      0.    ]
        #  [ 0.739   0.0095  0.0055  0.0035]
        #  [ 0.724   0.0325  0.012   0.0035]
        #  [ 0.7015  0.0515  0.0135  0.0075]
        #  [ 0.652   0.084   0.027   0.0135]
        #  [ 0.6115  0.116   0.0415  0.0155]]
        # [0.01, 0.1, 0.3, 0.5, 1.0, 1.5]
        def kin_random_restart(x, q_guess, sigma_add):
            num_restart = 0
            while not rospy.is_shutdown():
                q_try = q_guess + np.random.normal(0, sigma_add, 6)
                if kin.inverse(x, q_try) is not None:
                    return num_restart
                if num_restart == 15:
                    return num_restart
                num_restart += 1
            return -1
        def kin_random_progressive(x, q_guess):
            num_restart = 0
            sigma_adds = np.linspace(0.0001, np.pi, 16)
            while not rospy.is_shutdown():
                q_try = q_guess + np.random.normal(0, sigma_adds[num_restart], 6)
                if kin.inverse(x, q_try) is not None:
                    return num_restart
                if num_restart == 15:
                    return num_restart
                num_restart += 1
            return -1
        sigmas = [-1, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5]
        x_sigma = 0.1
        prcs = np.zeros((len(sigmas), 4))
        for j, sigma in enumerate(sigmas):
            rr_list = []
            n = 2000
            for i in range(n):
                q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                x_rand = kin.forward(q_rand)
                x_diff = np.mat(np.random.normal(0., x_sigma, 3)).T
                x_want = x_rand.copy()
                x_want[:3,3] += x_diff
                if sigma == -1:
                    rr_list.append(kin_random_progressive(x_want, q_rand))
                else:
                    rr_list.append(kin_random_restart(x_want, q_rand, sigma))
            d = defaultdict(int)
            for rr in rr_list:
                d[rr] += 1
            print d
            for i in range(4):
                prcs[j][i] = float(d[i]) / n
        print prcs
        print sigmas

    if False:
        def kin_random_progressive(x, q_guess, max_val):
            num_restart = 0
            sigma_adds = np.linspace(0.0001, max_val, 16)
            while not rospy.is_shutdown():
                if num_restart == 0:
                    q_try = q_guess
                else:
                    q_try = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                #q_try = q_guess + 2*(np.random.rand(6)-0.5)*sigma_adds[num_restart]
                if kin.inverse(x, q_try) is not None:
                    return num_restart
                if num_restart == 15:
                    return num_restart
                num_restart += 1
            return -1
        sigmas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        x_sigma = 0.1
        prcs = np.zeros((len(sigmas), 4))
        for j, sigma in enumerate(sigmas):
            rr_list = []
            n = 2000
            for i in range(n):
                q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                x_rand = kin.forward(q_rand)
                x_diff = np.mat(np.random.normal(0., x_sigma, 3)).T
                x_want = x_rand.copy()
                x_want[:3,3] += x_diff
                rr_list.append(kin_random_progressive(x_want, q_rand, sigma))
            d = defaultdict(int)
            for rr in rr_list:
                d[rr] += 1
            print d
            for i in range(4):
                prcs[j][i] = float(d[i]) / n
        print prcs
        print sigmas

    if False:
        def inverse_kin_search(x, q_guess, pos_var, rot_var):
            num_restart = 0
            while not rospy.is_shutdown():
                if num_restart == 0:
                    x_try = x
                else:
                    x_diff = PoseConv.to_homo_mat((2*(np.random.rand(3)-0.5)*pos_var).tolist(),
                                                  (2*(np.random.rand(3)-0.5)*rot_var).tolist())
                    x_try = x.copy()
                    x_try[:3,:3] = x_diff[:3,:3] * x_try[:3,:3]
                    x_try[:3,3] += x_diff[:3,3]
                #q_try = q_guess + 2*(np.random.rand(6)-0.5)*sigma_adds[num_restart]
                if kin.inverse(x_try, q_guess, restarts=1) is not None:
                    return num_restart
                if num_restart == 15:
                    return num_restart
                num_restart += 1
            return -1
        #sigmas = [0.00001, 0.005, 0.01, 0.02, 0.03]
        sigmas = np.array([0.00001, 0.05, 0.1, 0.166, 0.25, 0.333])*np.pi
        x_sigma = 0.1
        prcs = np.zeros((len(sigmas), 4))
        for j, sigma in enumerate(sigmas.tolist()):
            rr_list = []
            n = 1000
            for i in range(n):
                q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                x_rand = kin.forward(q_rand)
                x_diff = np.mat(np.random.normal(0., x_sigma, 3)).T
                x_want = x_rand.copy()
                x_want[:3,3] += x_diff
                rr_list.append(inverse_kin_search(x_want, q_rand, sigma/18, sigma))
            d = defaultdict(int)
            for rr in rr_list:
                d[rr] += 1
            print d
            for i in range(4):
                prcs[j][i] = float(d[i]) / n
        print prcs
        print sigmas

    if True:
        # inverse_rand_search can improve solution likelihood significantly
        # defaultdict(<type 'int'>, {False: 738, True: 262})
        # defaultdict(<type 'int'>, {False: 818, True: 182})
        # defaultdict(<type 'int'>, {False: 833, True: 167})
        # defaultdict(<type 'int'>, {False: 872, True: 128})
        # defaultdict(<type 'int'>, {False: 897, True: 103})
        # defaultdict(<type 'int'>, {False: 924, True: 76})
        # [[ 0.738  0.262  0.     0.   ]
        #  [ 0.818  0.182  0.     0.   ]
        #  [ 0.833  0.167  0.     0.   ]
        #  [ 0.872  0.128  0.     0.   ]
        #  [ 0.897  0.103  0.     0.   ]
        #  [ 0.924  0.076  0.     0.   ]]
        # [ 0.      0.1571  0.3142  0.5215  0.7854  1.0462]
        #  0.        ,  0.00872778,  0.01745556,  0.02897222,  0.04363333, 0.05812222
        #sigmas = [0.00001, 0.005, 0.01, 0.02, 0.03]
        sigmas = np.array([0.00001, 0.05, 0.1, 0.166, 0.25, 0.333])*np.pi
        x_sigma = 0.1
        prcs = np.zeros((len(sigmas), 4))
        for j, sigma in enumerate(sigmas.tolist()):
            rr_list = []
            n = 1000
            for i in range(n):
                q_rand = 2*(np.random.rand(6)-0.5)*(2*np.pi)
                x_rand = kin.forward(q_rand)
                x_diff = np.mat(np.random.normal(0., x_sigma, 3)).T
                x_want = x_rand.copy()
                x_want[:3,3] += x_diff
                rr_list.append(kin.inverse_rand_search(x_want, q_rand, sigma/18, sigma) is None)
            d = defaultdict(int)
            for rr in rr_list:
                d[rr] += 1
            print d
            for i in range(4):
                prcs[j][i] = float(d[i]) / n
        print prcs
        print sigmas

if __name__ == "__main__":
    main()
