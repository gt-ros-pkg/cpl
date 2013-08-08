import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
import numpy as np

class RBFController:
    '''
    Class to implement an RBF network controller based on PILCO learning for use in control of the robot.
    '''

    def feedbackControl(self, X):
        inp = np.zeros((self.N,self.D))
        for r in xrange(self.N):
            inp[r,:] = self.X_pi.T[r,:]-X.T

        k = None
        # Compute predicted mean
        U = np.zeros((self.E,1))
        for i in xrange(self.E):
            Lambda = self.lambdas[i] # NOTE: iL
            sf2 = self.sf2s[i] # NOTE: c
            sigmaBeta2 = self.sigmaBeta2s[i]
            t = np.matrix(inp)*Lambda # NOTE: in & t
            tb = np.sum(np.multiply(t,t),1)/2
            l = np.exp(-tb)
            lb = np.reshape(np.multiply(l.squeeze(),self.beta[:,i]),(self.N,1))
            U[i] = sf2*np.sum(lb)
            k_i = 2*self.Hyp[i,self.D]-tb;
            if k is None:
                k = k_i
            else:
                k = np.hstack([k, k_i])

        print 'U = ', U
        # Compute predicted covariance
        S = np.zeros((self.E, self.E))
        for i in xrange(self.E):
            ii = inp/np.exp(2.*self.Hyp[i,:self.D])
            for j in xrange(i+1):
                R = np.eye(self.D)
                ij = inp/np.exp(2.*self.Hyp[j,:self.D])
                L = np.asmatrix(np.exp(k[:,i]+k[:,j].T))
                A0 = np.asmatrix(self.beta[:,i])
                A1 = np.asmatrix(self.beta[:,j]).T
                print 'A0 = ', A0
                print 'A1 = ', A1
                S[i,j] = np.asmatrix(self.beta[:,i])*L*np.asmatrix(self.beta[:,j]).T
                S[j,i] = S[i,j]
            S[i,i] += 1e-6
        print 'U*U.T = ', U*U.T
        S = S - U*U.T
        print 'S = ', S
        # Keep values in range [-max_U, max_U]
        if self.max_U is not None:
            F = self.D+self.E
            j = range(self.D, F)
            M = np.vstack([X, U])
            V = np.zeros((F,F))
            V[np.ix_(j,j)] = S
            print 'M = ', M
            print 'V = ', V
            return self.gaussian_saturate(M, V, j, self.max_U)
        return U

    def computeBetaPi(self):
        '''
        Compute the RBF network weights in GP form
        '''
        self.beta = np.zeros((self.N, self.E))
        self.lambdas = []
        self.sf2s = []
        self.sigmaBeta2s = []
        for i in xrange(self.E):
            Lambda = np.diag(np.exp(-1*self.Hyp[i,:self.D])) # Length scale
            sf2 = np.exp(2*self.Hyp[i,self.D]) # signal variance
            sigmaBeta2 = np.exp(2*self.Hyp[i,self.D+1]) # kernel noise
            KX = np.zeros((self.N, self.N))
            # TODO: Make this batch
            for r in xrange(self.N):
                for c in range(i, self.N):
                    krc = self.squaredExponentialKernelARD(self.X_pi[:,r], self.X_pi[:,c], Lambda, sf2)
                    KX[r,c] = krc
                    KX[c,r] = krc
            A = KX+sigmaBeta2*np.eye(self.N)
            B = self.Y_pi[:,i]
            betai = np.linalg.solve(A, B)
            self.lambdas.append(Lambda)
            self.sf2s.append(sf2)
            self.sigmaBeta2s.append(sigmaBeta2)
            self.beta[:,i] = betai.squeeze()

    def squaredExponentialKernelARD(self, x, c, ell, sf2):
        '''
        Squared exponential with automatic relavence determination pre-computed parameters 'self.ell_diag'
        '''
        return sf2*np.exp(-self.squaredDist(ell*x, ell*c)/2)

    def squaredExponentialKernel(self, x, c, sf2):
        '''
        Squared exponential without the length scale hyperparemeters
        '''
        return sf2*np.exp(-self.squaredDist(x,c)/2)

    def squaredDist(self, x1, x2):
        '''
        Get the squared distance between two vectors
        TODO: Make this work for full matrices
        '''
        xx = x1-x2
        xx = np.multiply(xx,xx)
        xx = np.sum(xx)
        return xx

    def gaussian_saturate(self, m, v, i, e):
        m = np.asmatrix(m)
        d = len(m)
        I = len(i)
        i = np.hstack([i, np.asarray(i)+d])
        P = np.asmatrix(np.vstack([np.eye(d), 3.*np.eye(d)]))
        m = P*m
        e = np.asmatrix(np.hstack([9.*e, e])/8).T
        va = P*v*P.T
        va = (va+va.T)/2.
        vi = va[i][:,i]
        vii = np.asmatrix(np.diag(vi)).T
        mi = m[i,:]

        # Get mean
        M2 = np.multiply(np.multiply(e, np.exp(-vii/2)), np.sin(mi));
        # Combine back to correct dimensions
        P = np.asmatrix(np.hstack([np.eye(I), np.eye(I)]))
        print 'M2 = ', M2
        print 'P = ', P
        return P*M2

#
# I/O Functions
#
    def loadRBFController(self, controller_path):
        '''
        Read controller data from disk that was learned by PILCO
        '''
        M_LINE = 0
        N_LINE = 1
        E_LINE = 2
        D_LINE = 3
        MAX_U_LINE = 4
        HYP_LINE = 5
        TARGET_LINE = 6

        rospy.logwarn('controller_path:'+controller_path)
        controller_file = file(controller_path,'r')
        data_in = controller_file.readlines()
        controller_file.close()
        M = int(data_in[M_LINE].split()[1])
        N = int(data_in[N_LINE].split()[1])
        E = int(data_in[E_LINE].split()[1]) # Length of policy output
        D = int(data_in[D_LINE].split()[1]) # Length of policy input
        self.max_U = np.asarray([float(u) for u in data_in[MAX_U_LINE].split()[1:]])
        Hyp = np.asarray([float(h) for h in data_in[HYP_LINE].split()])
        self.Hyp = np.reshape(Hyp, (E, D+2))
        data_in = data_in[TARGET_LINE:]
        Y_pi = np.asmatrix([d.split() for d in data_in[:N]],'float')
        X_pi = np.asmatrix([d.split() for d in data_in[N:]],'float').T
        self.X_pi = X_pi
        self.Y_pi = Y_pi
        self.N = N
        self.D = D
        self.E = E
        # Precompute stuff
        self.computeBetaPi()
