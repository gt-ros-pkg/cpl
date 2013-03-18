#! /usr/bin/python

import numpy as np

def make_tri(h, q, qd_i=0., qdd_i=0., qd_f=0., qdd_f=0.):
    n = len(q)
    assert(n >= 4)
    a, b, c, d = np.zeros(n-2), np.zeros(n-2), np.zeros(n-2), np.zeros(n-2)
    a[0] = 0
    b[0] = 2*(h[0]+h[1])/h[1] + h[0]**2/h[1] * (1/h[0]+1/h[1])
    c[0] = 1
    d[0] = 6/h[1] * (q[2]/h[1]+q[0]/h[0]) - 6/h[1] * (1/h[0]+1/h[1]) * (q[0]+h[0]*qd_i+h[0]**2/3*qdd_i) - h[0]/h[1]*qdd_i
    a[1] = h[1]/h[2] - h[0]**2/(h[1]*h[2])
    b[1] = 2*(h[1]+h[2])/h[2]
    c[1] = 1
    d[1] = 6/(h[1]*h[2]) * (q[0] + h[0]*qd_i + h[0]**2/3*qdd_i) + 6*q[3]/h[2]**2 - 6/h[2] * (1/h[1]+1/h[2]) *q[2]
    for k in range(2,n-4):
        #%i = k+1
        a[k] = h[k]/h[k+1]
        b[k] = 2*(h[k]+h[k+1])/h[k+1]
        c[k] = 1
        d[k] = 6/h[k+1] * ((q[k+2]-q[k+1])/h[k+1]-(q[k+1]-q[k])/h[k])
    a[n-4] = h[n-4]/h[n-3]
    b[n-4] = 2*(h[n-4]+h[n-3])/h[n-3]
    c[n-4] = 1 - h[n-2]**2/h[n-3]**2
    d[n-4] = 6/h[n-3]**2 * (q[n-1] - qd_f*h[n-2] + h[n-2]**2/3*qdd_f) - 6/h[n-3] * (1/h[n-3]+1/h[n-4]) * q[n-3] + 6/(h[n-3]*h[n-4]) * q[n-4]
    a[n-3] = h[n-3]/h[n-2]
    b[n-3] = 2*(h[n-2]+h[n-3])/h[n-2]+(1/h[n-2]+1/h[n-3])*h[n-2]
    c[n-3] = 0
    d[n-3] = -6/h[n-2]*(1/h[n-2]+1/h[n-3])*(q[n-1]-qd_f*h[n-2]+h[n-2]**2/3*qdd_f)+6*q[n-1]/h[n-2]**2+6*q[n-3]/(h[n-2]*h[n-3])-qdd_f
    return a, b, c, d

def TDMASolve(a, b, c, d):
    # from Wikipedia
    n = len(d) # n is the numbers of rows, a and c has length n-1
 
    c[0] /= b[0] # division by zero risk.
    d[0] /= b[0]
 
    for i in xrange(1,n-1):
        divby = b[i] - (a[i] * c[i-1])
        c[i] /= divby
        d[i] = (d[i] - a[i] * d[i-1]) / divby
    d[n-1] = (d[n-1] - a[n-1] * d[n-2])/( b[n-1] - a[n-1] * c[n-2])
 
    # back substitution
    x = [0]*n
    x[n-1] = d[n-1]
    for i in reversed(xrange(n-1)):
        x[i] = d[i] - c[i] * x[i+1]
    return x

class CubicSpline(object):
    def __init__(self, tk, q, qd, qdd):
        self.tk = tk
        self.q = q
        self.qd = qd
        self.qdd = qdd
        self.h = tk[1:] - tk[:-1]
        self.A = qdd[:-1]/self.h
        self.B = qdd[1:]/self.h
        self.C = qd[:-1] + qdd[:-1]/2*self.h
        self.D = q[:-1] - self.A/6*self.h**3 - self.C*tk[:-1]

    def sample(self, t):
        tk = self.tk
        if abs(t-tk[0]) < 0.0001:
            t = tk[0]
        k = np.argmax(tk>t)-1
        if abs(t-tk[-1]) < 0.0001:
            k = len(tk)-2
        if t < tk[0] or t > tk[-1]:
            return None, None, None
        A, B, C, D = self.A[k], self.B[k], self.C[k], self.D[k]
        dtA = tk[k+1]-t
        dtB = t-tk[k]
        y = A/6*dtA**3 + B/6*dtB**3 + C*t + D
        yd = -A/2*dtA**2 + B/2*dtB**2 + C
        ydd = A*dtA + B*dtB
        return y, yd, ydd
    
    def view(self):
        y, yd, ydd = np.zeros(3000), np.zeros(3000), np.zeros(3000)
        ts = np.linspace(self.tk[0],self.tk[-1],3000)
        for i, t in enumerate(ts):
            y[i], yd[i], ydd[i] = self.sample(t)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.subplot(311)
        plt.plot(ts,ydd)
        plt.plot(self.tk,self.qdd,'go')
        plt.subplot(312)
        plt.plot(ts,yd)
        plt.plot(self.tk,self.qd,'go')
        plt.subplot(313)
        plt.plot(ts,y)
        plt.plot(self.tk,self.q,'go')
        plt.show()

    @staticmethod
    def generate(t, q, qd_i=0., qd_f=0., qdd_i=0., qdd_f=0.):
        N = len(t)+1
        # add in knot placeholders
        t = np.array([t[0]] + [(t[0]+t[1])/2.] + t[1:-1] + [(t[-2]+t[-1])/2.] + [t[-1]]).copy() 
        h = np.array(t[1:])-np.array(t[0:-1])
        q = np.array([q[0]] + [0] + q[1:-1] + [0] + [q[-1]]).copy() # add in knot placeholders
        a, b, c, d = make_tri(h, q)
        x = TDMASolve(a,b,c,d)
        qdd = np.array([qdd_i] + x + [qdd_f])
        C, D = np.zeros(N), np.zeros(N)
        for k in range(N):
            if k == 0:
                C[0] = qd_i + qdd_i*h[0]/2
            else:
                C[k] = qdd[k]*(h[k]+h[k-1])/2 + C[k-1]
            if k == 1:
                q[1] = qdd[1]/6*h[0]**2 + C[0]*t[1] + D[0]
            if k == N-1:
                C[k] = qd_f-qdd[k+1]/2*h[k]
                D[k] = q[k+1] - qdd[k+1]/6*h[k]**2 - C[k]*t[k+1]
                q[k] = qdd[k]/6*h[k]**2 + C[k]*t[k+1] + D[k]
            else:
                D[k] = q[k] - qdd[k]/6*h[k]**2 - C[k]*t[k]
        qd = np.append(C-qdd[:-1]/2*h, qd_f)
        return CubicSpline(np.array(t), q, qd, qdd)

def main():
    def rand_waypts(num_pts):
        t = [0] + np.cumsum(2*np.random.rand(num_pts-1)+0.5).tolist()
        q = [0] + (2*(np.random.rand(num_pts-2)-0.5)*np.pi*2).tolist() + [np.pi]
        return t, q
    if False:
        import time
        start_time = time.time()
        num_gen, period = 0, 10.
        while time.time() - start_time < period:
            t, q = rand_waypts(400)
            cs = CubicSpline.generate(t,q)
            num_gen += 1
        print "Generation time: %1.6f, Rate: %d" % (period / num_gen, int(num_gen / period))
    if True:
        t, q = rand_waypts(10)
        cs = CubicSpline.generate(t,q)
        cs.view()

if __name__ == "__main__":
    main()
