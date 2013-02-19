import numpy as np

class RBFControl:
    '''
    Class to implement an RBF network controller based on PILCO learning for use in control of the robot.
    '''

    def loadRBFController(self, controller_path):
        '''
        Read controller data from disk that was learned by PILCO
        '''
        controller_file = file(controller_path,'r')
        data_in = controller_file.readlines()
        controller_file.close()
        M = int(data_in[0].split()[1])
        N = int(data_in[1].split()[1])
        Un = int(data_in[2].split()[1])
        D = int(data_in[3].split()[1])
        Hyp = np.asarray([float(h) for h in data_in[4].split()])
        self.ell = np.exp(Hyp[:D]) # characteristic length scale
        self.ell_diag = np.diag(1/self.ell)
        self.sf2 = np.exp(2*Hyp[D+1]) # signal variance

        data_in = data_in[5:]
        Ypi = np.asmatrix([d.split() for d in data_in[:N]],'float')
        Xpi = np.asmatrix([d.split() for d in data_in[N:]],'float')
        self.Xpi = Xpi.T
        self.Ypi = Ypi.T
        self.N = N
        self.D = D

    def computeBetaPi(self):
        '''
        Compute the RBF network weights in GP form
        '''
        self.beta = np.zeros((self.N,self.N))
        KX = np.zeros((self.N,self.N))
        for i in xrange(self.N):
            for j in range(i, self.N):
                xk = self.squaredExponentialKernelARD(self.Xpi[:,i], self.Xpi[:,j])
                KX[i,j] = xk
                KX[j,i] = xk
        self.beta = KX
        print self.beta

    def squaredExponentialKernelARD(self, x, c):
        '''
        Squared exponential with automatic relavence determination pre-computed parameters 'self.ell_diag'
        '''
        r = self.squaredDist(self.ell_diag*x,self.ell_diag*c)
        return self.sf2*np.exp(r/2)

    def squaredExponentialKernel(self, x, c, sf):
        '''
        Squared exponential without the length scale hyperparemeters
        '''
        r = self.squaredDist(x,c)
        return sf*np.exp(r/2)

    def squaredDist(self, x1, x2):
        '''
        Get the squared distance between two vectors
        TODO: Make this work for full matrices
        '''
        xx = x1-x2
        xx = np.multiply(xx,xx)
        xx = np.sum(xx)
        return xx
