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
            inp[r,:] = self.Xpi.T[r,:]-X.T
        U = np.zeros((self.E,1))
        for i in xrange(self.E):
            Lambda = self.lambdas[i]
            sf2 = self.sf2s[i]
            sigmaBeta2 = self.sigmaBeta2s[i]
            t = np.matrix(inp)*Lambda
            # Note not diving since we aren't using variance here
            # B = Lambda*V*Lambda+np.eye(self.D)
            # t = np.linalg.solve(B.T,t.T)
            l = np.exp(-np.sum(np.multiply(t,t),1)/2)
            lb = np.reshape(np.multiply(l.squeeze(),self.beta[:,i]),(self.N,1))
            U[i] = sf2*np.sum(lb)
        if self.max_U is not None:
            F = self.D+self.E
            j = range(D,F)
            i = range(0,D)
            M = np.vstack([X, U])
            # TODO: get variance
            v = np.zeros((F,F))
            return self.saturate(M, v, i, self.max_U)
        return U

    def loadRBFController(self, controller_path):
        '''
        Read controller data from disk that was learned by PILCO
        '''
        rospy.logwarn('controller_path:'+controller_path)
        controller_file = file(controller_path,'r')
        data_in = controller_file.readlines()
        controller_file.close()
        M = int(data_in[0].split()[1])
        N = int(data_in[1].split()[1])
        E = int(data_in[2].split()[1]) # Length of policy output
        D = int(data_in[3].split()[1]) # Length of policy input
        Hyp = np.asarray([float(h) for h in data_in[4].split()])
        self.Hyp = np.reshape(Hyp, (E, D+2))
        data_in = data_in[5:]
        Ypi = np.asmatrix([d.split() for d in data_in[:N]],'float')
        Xpi = np.asmatrix([d.split() for d in data_in[N:]],'float')
        self.Xpi = Xpi.T
        self.Ypi = Ypi
        self.N = N
        self.D = D
        self.E = E

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
                    krc = self.squaredExponentialKernelARD(self.Xpi[:,r], self.Xpi[:,c], Lambda, sf2)
                    KX[r,c] = krc
                    KX[c,r] = krc
            A = KX+sigmaBeta2*np.eye(self.N)
            B = self.Ypi[:,i]
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

    def saturate(self, m, v, i, e):
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
        return P*M2
