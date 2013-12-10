#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2013, Georgia Institute of Technology
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#  * Neither the name of the Georgia Institute of Technology nor the names of
#     its contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

from math import sqrt
from numpy import finfo
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plotter

def get_x_u_from_q(q, x0, H, n, m):
    x = [x0]
    u = []
    step = m+n
    score = 0
    for k in xrange(H):
        u_start = k*step
        u_stop = u_start + m
        x_start = u_stop
        x_stop = x_start + n
        u.append(np.asarray(q[u_start:u_stop]))
        x.append(np.asarray(q[x_start:x_stop]))
    return (x,u)

def pushMPCObjectiveFunction(q, H, n, m, x0, x_d, xtra, dyn_model):
    '''
    q - decision variable vector of the form (U[0], X[1], U[1],..., U[H-1], X[H])
    H - time horizon
    n - system state dimension
    m - control dimension
    x0 - inital state
    x_d - desired trajectory
    xtra - other features for dynamics prediction
    dyn_model - model simulating system dynamics of the form x[k+1] = dyn_model.predict(x[k], u[k], xtra)
    '''
    step = m+n
    x_d_length = len(x_d[0])
    cost = 0
    for k in xrange(H):
        x_start = m+k*step
        x_stop = x_start+x_d_length
        # Pull out x from q
        x_k_plus_1 = np.asarray(q[x_start:x_stop])
        # Compute sum of squared error on current trajectory time step
        cost = sum((x_k_plus_1 - x_d[k+1])**2)
    return cost

def pushMPCObjectiveGradient(q, H, n, m, x0, x_d, xtra, dyn_model):
    gradient = np.zeros(len(q))
    step = m+n
    x_d_length = len(x_d[0])
    score = 0
    for k in xrange(H):
        x_i_start = m+k*step
        for j, i in enumerate(range(x_i_start, x_i_start+x_d_length)):
            gradient[i] = 2.0*(q[i]-x_d[k+1][j])
    return gradient

def pushMPCConstraints(q, H, n, m, x0, x_d, xtra, dyn_model):
    x, u = get_x_u_from_q(q, x0, H, n, m)
    G = []
    for k in xrange(H):
        G.extend(dyn_model.predict(x[k], u[k], xtra) - x[k+1])
    return np.array(G)

def pushMPCConstraintsGradients(q, H, n, m, x0, x_d, xtra, dyn_model):
    x, u = get_x_u_from_q(q, x0, H, n, m)

    # Build Jacobian of the constraints
    num_constraints = H*n
    J = np.zeros((num_constraints, len(q)))

    # Ignore partials for x0 on the initial time step, since this is fixed
    J_f_k0 = dyn_model.jacobian(x[0], u[0])
    k_plus_1_constraint = np.eye(n)*-1.0
    J[0:n, 0:m] = J_f_k0[:,n:n+m]
    J[0:n, m:m+n] = k_plus_1_constraint

    # Setup partials for x[1], u[1], ..., x[H-1], u[H-1], x[H]
    for k in range(1,H):
        row_start = k*n
        row_stop = row_start+n
        col_start = m+(n+m)*(k-1)
        col_stop = col_start+m+n
        # Each of the H row blocks have the form f(x[k], u[k]) - x[k+1] (n dimensions)
        # Get jacobian for f(x[k], u[k])
        J_f_k = dyn_model.jacobian(x[k], u[k])
        J[row_start:row_stop, col_start:col_stop] = J_f_k
        # Set derivative for x[k+1]
        col_start = col_stop
        col_stop = col_start + n
        J[row_start:row_stop, col_start:col_stop] = k_plus_1_constraint

    return J

class ModelPredictiveController:
    def __init__(self, model, H=5, u_max=1.0, delta_t=1.0):
        '''
        model - prediction function for use inside the optimizer
        H - the lookahead horizon for MPC
        u_max - maximum allowed velocity
        '''
        self.init_from_previous = False
        self.dyn_model = model
        self.H = H # Time horizon
        self.n = 5 # Predicted state space dimension
        self.m = 2 # Control space dimension
        self.u_max = u_max
        self.delta_t = delta_t
        self.max_iter = 1000 # Max number of iterations
        self.ftol = 1.0E-5 # Accuracy of answer
        self.epsilon = sqrt(finfo(float).eps)
        self.opt_options = {'iter':self.max_iter,
                            'acc':self.ftol,
                            'iprint':1,
                            'epsilon':self.epsilon,
                            'full_output':True}
        self.regenerate_bounds()

    def regenerate_bounds(self):
        bounds_k = []
        for i in xrange(self.m):
            bounds_k.append((-self.u_max, self.u_max))
        for i in xrange(self.n):
            bounds_k.append((-1.0E12,1.0E12))
        self.opt_bounds = []
        for i in xrange(self.H):
            self.opt_bounds.extend(bounds_k)
        self.opt_options['bounds'] = self.opt_bounds

    def feedbackControl(self, x0, x_d, xtra = []):
        if self.init_from_previous:
            q0 = self.init_q0_from_previous(x_d, xtra)
        else:
            U_init = self.get_U_init(x0, x_d)
            q0 = self.get_q0(x0, U_init, xtra)

        opt_args = (self.H, self.n, self.m, x0, x_d, xtra, self.dyn_model)
        # Perform optimization
        res = opt.fmin_slsqp(pushMPCObjectiveFunction, q0, fprime = pushMPCObjectiveGradient,
                             f_eqcons = pushMPCConstraints, fprime_eqcons = pushMPCConstraintsGradients,
                             args = opt_args, **self.opt_options)
        q_star = res[0]
        self.q_star_prev = q_star[:]
        opt_val = res[1]

        return q_star

    def get_U_init(self, x0, x_d):
        # TODO: Get initial guess at U from cur_state and trajectory (or at least goal)
        U_init = []
        for k in xrange(self.H):
            if k % 2 == 0:
                u_x = self.u_max*0.01
            else:
                u_x = self.u_max*0.4
            if k / 3 == 0:
                u_y = self.u_max*0.2
            elif k / 3 == 1:
                u_y = -self.u_max*0.6
            else:
                u_y = self.u_max*0.5
            U_init.append(np.array([u_x, u_y]))
        return U_init

    def get_q0(self, x0, U, xtra):
        q0 = []
        x_k = x0
        for k in xrange(self.H):
            u_k = U[k]
            x_k_plus_1 = self.dyn_model.predict(x_k, u_k, xtra)
            for u_i in u_k:
                q0.append(u_i)
            for x_i in x_k_plus_1:
                q0.append(x_i)
            x_k = x_k_plus_1
        return np.array(q0)

    def init_q0_from_previous(self, x_d, xtra):
        # Remove initial control and loc from previous solution
        N = self.n+self.m
        q0 = self.q_star_prev[N:]
        k0 = len(q0)/N
        N_to_add = self.H - k0
        # Add the necessary number of more controls and locs to get to H tuples
        if N_to_add > 0:
            for k in xrange(N_to_add):
                x_k = q0[-self.n:]
                # Initialize next control assuming straight line motion between via points
                deltaX = x_d[k0+k+1] - x_d[k0+k]
                next_u = np.array([deltaX[0]/self.delta_t, deltaX[1]/self.delta_t])
                q0 = np.concatenate([q0, next_u])
                # Use dynamics to add next location
                q0 = np.concatenate([q0, self.dyn_model.predict(x_k, next_u, xtra)])
        else:
            # Remove the necessary number of more controls and locs to get to H tuples
            q0 = q0[:N*self.H]
        return np.array(q0)

def plot_desired_vs_controlled(q_star, X_d, x0, n, m, show_plot=True, suffix='', t=0, out_path=''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)
    plotter.figure()
    # Plot desired
    x_d = [X_d_k[0] for X_d_k in X_d]
    y_d = [X_d_k[1] for X_d_k in X_d]
    theta_d = [X_d_k[2] for X_d_k in X_d]
    plotter.plot(x_d, y_d, 'r')
    plotter.plot(x_d, y_d, 'ro')

    # Plot predicted
    x_hat = [X_k[0] for X_k in X[t:]]
    y_hat = [X_k[1] for X_k in X[t:]]
    theta_hat = [X_k[1] for X_k in X[t:]]
    plotter.plot(x_hat, y_hat,'b')
    plotter.plot(x_hat, y_hat,'b+')

    # Plot observed / GT
    x_gt = [X_k[0] for X_k in X[:t+1]]
    y_gt = [X_k[1] for X_k in X[:t+1]]
    theta_gt = [X_k[1] for X_k in X[:t+1]]
    plotter.plot(x_gt, y_gt,'g')
    plotter.plot(x_gt, y_gt,'g+')

    plot_title = 'MPC_Push_Control_Trajectory'+suffix
    plotter.title(plot_title)
    plotter.xlabel('x (meters)')
    plotter.ylabel('y (meters)')
    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

def plot_controls(q_star, X_d, x0, n, m, u_max, show_plot=True, suffix='', t=0, out_path=''):
    H = len(q_star)/(n+m)
    X,U =  get_x_u_from_q(q_star, x0, H, n, m)

    Ux = [u_k[0] for u_k in U]
    Uy = [u_k[1] for u_k in U]

    plotter.figure()
    plotter.plot(Ux, 'r')
    plotter.plot(Ux, 'ro')

    plotter.plot(Uy, 'b')
    plotter.plot(Uy, 'b+')
    plotter.plot(np.ones(H)*u_max, 'g')
    plotter.plot(np.ones(H)*-u_max, 'g')

    plot_title = 'MPC_Push_Control_Inputs'+suffix
    plotter.title(plot_title)
    ax = plotter.gca()
    ax.set_ylim(-1.1*u_max, 1.1*u_max)
    plotter.xlabel('Time Step')
    plotter.ylabel('U (meters/sec)')
    if len(out_path) > 0:
        plotter.savefig(out_path+plot_title+'.png')
    if show_plot:
        plotter.show()

class MPCSolutionIO:
    def __init__(self):
        self.out_file = None
        self.out_lines = []

    def open_out_file(self, file_name):
        self.out_file = file(file_name, 'a')

    def close_out_file(self):
        if self.out_file is not None:
            self.out_file.close()

    def buffer_line(self, k0, q):
        out_line = self.generate_line(k0, q)
        self.out_lines.append(out_line)

    def write_line(self, k0, q):
        out_line = self.generate_line(k0, q)
        self.out_file.write(out_line)

    def write_buffer_to_disk(self):
        for line in self.out_lines:
            self.out_file.write(line)
        self.out_file.flush()
        self.out_lines = []

    def generate_line(self, k0, q):
        out_line = str(k0) + ' ' + str(q)
        out_line += '\n'
        return out_line

    def parse_line(self, line):
        elems = line.split()
        k0 = int(elems[0])
        q = [float(x) for x in elems[1:]]
        return (k0, q)

    def read_file(self, file_name):
        in_file = file(file_name, 'r')
        data = [self.parse_line(l) for l in in_file.readlines()]
        in_file.close()
        return data
