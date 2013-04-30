import numpy as np
import copy
from numpy import sqrt, sin, cos, arctan2, arccos, arcsin, abs, pi
import roslib
#roslib.load_manifest('ur_cart_move')
roslib.load_manifest('hrl_geom')
import rospy
from hrl_geom.pose_converter import PoseConv
from geometry_msgs.msg import PoseStamped
from hrl_geom.transformations import rotation_from_matrix as mat_to_ang_axis_point

d1, a2, a3, d4, d5, d6 = [0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922]
UR10_A = [0, a2, a3, 0, 0, 0]
UR10_D = [d1, 0, 0, d4, d5, d6]
UR10_L = [pi/2, 0, 0, pi/2, -pi/2, 0]

#def inverse_rrr(T04, a=[a1, a2, a3], d=[d1, d2, d3]):
def inverse_rrr(p04x, p04y, x04x, x04y, a, d):
    a1, a2, a3 = a
    d1, d2, d3 = d
    #p04x, p04y = T04[0,3], T04[1,3]
    #x04x = T04[0,0]
    #x04y = T04[1,0]
    qs1 = []
    p13x = p04x - a3*x04x
    p13y = p04y - a3*x04y
    c2 = (p13x**2 + p13y**2 - a1**2 - a2**2) / (2.*a1*a2)
    if abs(abs(c2) - 1.) < ZERO_THRESH:
        c2 = np.sign(c2)
    elif abs(c2) > 1.:
        #print 'low c2'
        #print c2, p13x, p13y
        return []
    #print 'eeeeeeee', (p13x**2 + p13y**2 - a1**2 - a2**2), (2.*a1*a2)
    #print 'mmmmmmmm', c2, p13x, p13y
    for ssign in [1., -1.]:
        q = [0.]*3
        s2 = ssign*np.sqrt(1. - c2**2)
        q[1] = np.arctan2(s2, c2)
        if q[1] < 0.:
            q[1] += 2.*pi
        denom = a1**2 + a2**2 + 2*a1*a2*c2
        c1 = ((a1 + a2*c2) * p13x + a2*s2*p13y) / denom
        s1 = ((a1 + a2*c2) * p13y - a2*s2*p13x) / denom
        q[0] = np.arctan2(s1, c1)
        c12 = np.cos(q[0]+q[1])
        s12 = np.sin(q[0]+q[1])
        c3 = x04x*c12 + x04y*s12
        s3 = c12*x04y - s12*x04x
        q[2] = np.arctan2(s3, c3)
        qs1.append(q)
    #qs2 = []
    #for q in qs1:
    #    qs2.append(q)
    #    if q[0] > 0:
    #        qs2.append([q[0]-2.*pi, q[1], q[2]])
    #    else:
    #        qs2.append([q[0]+2.*pi, q[1], q[2]])
    #qs3 = []
    #for q in qs2:
    #    qs3.append(q)
    #    if q[1] > 0:
    #        qs3.append([q[0], q[1]-2.*pi, q[2]])
    #    else:
    #        qs3.append([q[0], q[1]+2.*pi, q[2]])
    #qs4 = []
    #for q in qs3:
    #    qs4.append(q)
    #    if q[2] > 0:
    #        qs4.append([q[0], q[1], q[2]-2.*pi])
    #    else:
    #        qs4.append([q[0], q[1], q[2]+2.*pi])
    #return qs4
    return qs1

def forward_rrr(q):
    T01 = np.mat([[np.cos(q[0]), -np.sin(q[0]), 0., 0.],
                  [np.sin(q[0]), np.cos(q[0]), 0., 0.],
                  [0., 0., 1., d1],
                  [0., 0., 0., 1.]])
    T12 = np.mat([[np.cos(q[1]), -np.sin(q[1]), 0., a1],
                  [np.sin(q[1]), np.cos(q[1]), 0., 0.],
                  [0., 0., 1., d2],
                  [0., 0., 0., 1.]])
    T23 = np.mat([[np.cos(q[2]), -np.sin(q[2]), 0., a2],
                  [np.sin(q[2]), np.cos(q[2]), 0., 0.],
                  [0., 0., 1., d3],
                  [0., 0., 0., 1.]])
    T34 = np.mat([[1., 0., 0., a3],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]])
    return T01*T12*T23*T34

def inv_mat(T):
    R = np.mat(np.eye(4))
    R[:3,:3] = T[:3,:3].T
    R[:3,3] = -T[:3,:3].T * T[:3,3]
    return R

Tb0 = np.mat([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T6e = np.mat([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def forward_kin(q, a, d, l, inner=False):
    Ts = []
    if inner:
        Tlast = np.mat(np.eye(4))
    else:
        Tlast = Tb0
    for qi, ai, di, li in zip(q, a, d, l):
        cqi, sqi = cos(qi), sin(qi)
        cli, sli = cos(li), sin(li)
        T = np.mat([[cqi, -sqi*cli,  sqi*sli, ai*cqi],
                    [sqi,  cqi*cli, -cqi*sli, ai*sqi],
                    [      0,          sli,          cli,         di],
                    [      0,                0,                0,          1]])
        Tlast = Tlast*T
        Ts.append(T)
    if not inner:
        Tlast *= T6e
    return Tlast, Ts

def forward_kin_fast(q, a, d, l):
    T = np.mat(np.eye(4))
    c1, c2, c3, c4, c5, c6 = cos(q[0]), cos(q[1]), cos(q[2]), cos(q[3]), cos(q[4]), cos(q[5])
    s1, s2, s3, s4, s5, s6 = sin(q[0]), sin(q[1]), sin(q[2]), sin(q[3]), sin(q[4]), sin(q[5])
    c234, s234 = cos(q[1]+q[2]+q[3]), sin(q[1]+q[2]+q[3])
    d1, a2, a3, d4, d5, d6 = d[0], a[1], a[2], d[3], d[4], d[5]
    T[0,0] = ((c1*c234-s1*s234)*s5)/2 - c5*s1 + ((c1*c234+s1*s234)*s5)/2
    T[0,1] = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2 + ((c1*c234+s1*s234)*c5)/2) - 
              (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2)
    T[0,2] = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2 - 
              s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2 + ((c1*c234+s1*s234)*c5)/2))
    T[0,3] = ((d5*(s1*c234-c1*s234))/2 - (d5*(s1*c234+c1*s234))/2 - 
              d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2 + (d6*(c1*c234+s1*s234)*s5)/2 - 
              a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3)
    T[1,0] = c1*c5 + ((s1*c234+c1*s234)*s5)/2 + ((s1*c234-c1*s234)*s5)/2
    T[1,1] = (c6*(((s1*c234+c1*s234)*c5)/2 - c1*s5 + ((s1*c234-c1*s234)*c5)/2) + 
              s6*((c1*c234-s1*s234)/2 - (c1*c234+s1*s234)/2))
    T[1,2] = (c6*((c1*c234-s1*s234)/2 - (c1*c234+s1*s234)/2) - 
              s6*(((s1*c234+c1*s234)*c5)/2 - c1*s5 + ((s1*c234-c1*s234)*c5)/2))
    T[1,3] = ((d5*(c1*c234-s1*s234))/2 - (d5*(c1*c234+s1*s234))/2 + d4*c1 + 
              (d6*(s1*c234+c1*s234)*s5)/2 + (d6*(s1*c234-c1*s234)*s5)/2 + d6*c1*c5 - 
              a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3)
    T[2,0] = ((c234*c5-s234*s5)/2 - (c234*c5+s234*s5)/2)
    T[2,1] = ((s234*c6-c234*s6)/2 - (s234*c6+c234*s6)/2 - s234*c5*c6)
    T[2,2] = (s234*c5*s6 - (c234*c6+s234*s6)/2 - (c234*c6-s234*s6)/2)
    T[2,3] = (d1 + (d6*(c234*c5-s234*s5))/2 + a3*(s2*c3+c2*s3) + a2*s2 - 
              (d6*(c234*c5+s234*s5))/2 - d5*c234)
    return T

ZERO_THRESH = 0.00001

def inverse_kin_old(T06, a, d, l, q6_des, debug=False):
    qs1 = []
    T = inv_mat(Tb0) * T06 * inv_mat(T6e)
    A = d[5]*T[1,2] - T[1,3]
    B = d[5]*T[0,2] - T[0,3]
    R = A*A + B*B
    if abs(A) < ZERO_THRESH:
        print '1: A low'
        return []
    elif abs(B) < ZERO_THRESH:
        print '1: B low'
        return []
    elif d[3]*d[3] > R:
        #print '1: Impossible solution'
        return []
    else:
        for i in range(2):
            qs1.append([0.]*6)
        acos = arccos(d[3] / sqrt(R)) 
        atan = arctan2(B, A)
        pos = acos - atan
        neg = -acos - atan
        if pos >= 0.:
            qs1[0][0] = pos
        else:
            qs1[0][0] = 2.*pi + pos
        if neg >= 0.:
            qs1[1][0] = neg
        else:
            qs1[1][0] = 2.*pi + neg
        #if pos < 0:
        #    qs1[2][0] = pos + 2.*pi
        #else:
        #    qs1[2][0] = pos - 2.*pi
        #if neg < 0:
        #    qs1[3][0] = neg + 2.*pi
        #else:
        #    qs1[3][0] = neg - 2.*pi
    qs2 = []
    for i in range(len(qs1)):
        for j in range(2):
            qs2.append(copy.copy(qs1[i]))
        if debug:
            print 'h', T[0,2]*sin(qs1[i][0]) - T[1,2]*cos(qs1[i][0])
            print 'h2', T
            acos = arccos(T[0,2]*sin(qs1[i][0]) - T[1,2]*cos(qs1[i][0]))
            print 'h3', qs1[i][0]
            print 'h2', (T[0,3]*sin(qs1[i][0]) - T[1,3]*cos(qs1[i][0])-d[3])/d[5]
        numer = (T[0,3]*sin(qs1[i][0]) - T[1,3]*cos(qs1[i][0])-d[3])
        if abs(abs(numer) - abs(d[5])) < ZERO_THRESH:
            div = np.sign(numer) * np.sign(d[5])
        else:
            div = numer / d[5]
        acos = arccos(div)
        if acos >= 0.:
            qs2[i*2+0][4] = acos
            qs2[i*2+1][4] = 2.*pi-acos
        else:
            qs2[i*2+0][4] = -acos
            qs2[i*2+1][4] = 2.*pi+acos

    qs3 = []
    for i in range(len(qs2)):
        for j in range(2):
            qs3.append(copy.copy(qs2[i]))
        s4 = sin(qs2[i][4])
        #print 's4', s4
        #print 'h2', (T[0,0]*sin(qs2[i][0]) - T[1,0]*cos(qs2[i][0]))
        #c1, s1 = cos(qs2[i][0]), sin(qs2[i][0])
        #acos = arctan2(-(T[1,1]*c1-T[0,1]*s1), T[1,0]*c1-T[0,0]*s1)
        #acos = np.arctan(T[2,0]/ T[2,1])
        #print 'k', acos
        #print 'k2', acos
        #acos = ( (-1.)**(i%2+0)* np.sign(T[2,2])**2 *pi/2.
        #        +(-1.)**2* np.sign(T[2,2])**2 *arcsin(T[1,0]) 
        #        +(-1.)**2* np.sign(T[2,2])**2 *qs2[i][0])
        if abs(s4) < ZERO_THRESH:
            #print '6: s4 low'
            qs3[i][5] = q6_des
            qs3[i+1][5] = q6_des + pi
        elif abs(abs(s4) - 1.) < ZERO_THRESH:
            acos = (-1.)**(i%2) * pi/2. + arcsin(T[1,0]) + qs2[i][0]
            if acos >= 0.:
                if T[2,2] >= 0.:
                    qs3[i*2+0][5] = 2.*pi-acos
                    qs3[i*2+1][5] = 2.*pi-acos
                else:
                    qs3[i*2+0][5] = acos
                    qs3[i*2+1][5] = acos
            else:
                if T[2,2] >= 0.:
                    qs3[i*2+0][5] = -acos
                    qs3[i*2+1][5] = -acos
                else:
                    qs3[i*2+0][5] = 2.*pi+acos
                    qs3[i*2+1][5] = 2.*pi+acos
        else:
            numer = (T[0,0]*sin(qs2[i][0]) - T[1,0]*cos(qs2[i][0]))
            if abs(abs(numer) - abs(s4)) < ZERO_THRESH:
                div = np.sign(numer) * np.sign(s4)
            else:
                div = numer / s4
            acos = arccos(div)
        #if abs(cos(acos-qs2[i][0])) - abs(T[0,0]) > ZERO_THRESH:
        #    acos += pi
        #if qs2[0][0] < pi and T[2,2] < 0.:
        #    acos -= pi
            if acos >= 0.:
                #if T[2,2] >= 0.:
                #    qs3[i*1+0][5] = 2.*pi-acos
                #else:
                #    qs3[i*1+0][5] = acos
                qs3[i*2+0][5] = 2.*pi-acos
                qs3[i*2+1][5] = acos
            else:
                #if T[2,2] >= 0.:
                #    qs3[i*1+0][5] = -acos
                #else:
                #    qs3[i*1+0][5] = 2.*pi+acos
                qs3[i*2+0][5] = -acos
                qs3[i*2+1][5] = 2.*pi+acos
        #print 'ssss', s4, qs3[i*1+0][5], qs3[i*1+1][5]
        
        #print '1111111111111', cos(qs3[i][5])*sin(qs3[i][0])*sin(qs3[i][4]) - cos(qs3[i][0])*sin(qs3[i][5]),  cos(qs3[i][5])*sin(qs3[i][0])*sin(qs3[i][4]) + cos(qs3[i][0])*sin(qs3[i][5]), T[0,0], T[1,0]
        for k in [0, 1]:
            if abs(abs(s4) - 1.) < ZERO_THRESH:
                tmp1 = cos(qs3[2*i+k][5])*sin(qs3[2*i+k][0])*sin(qs3[2*i+k][4])
                tmp2 = cos(qs3[2*i+k][0])*sin(qs3[2*i+k][5])
                #print sin(qs3[2*i+k][4])
                if abs(abs(tmp1 - tmp2) - abs(T[0,0])) < ZERO_THRESH:
                    if np.sign(tmp1 - tmp2) != np.sign(T[0,0]):
                        #qs3[2*i+k][5] -= pi
                        #qs3[2*i+k][0] *= -1
                        #qs3[i][5] *= -1
                        if sin(qs3[2*i+k][4]) > 0:
                            qs3[2*i+k][5] = -qs3[2*i+k][5] + 2*qs3[2*i+k][0]
                        else:
                            qs3[2*i+k][5] = -qs3[2*i+k][5] - 2*qs3[2*i+k][0]
                        #print tmp1 - tmp2
                        #print T[0,0]
                        #print 'yo1'
                else:
                    if np.sign(tmp1 + tmp2) != np.sign(T[0,0]):
                        #qs3[i][5] -= pi
                        #qs3[i][0] *= -1
                        #qs3[i][5] *= -1
                        if sin(qs3[2*i+k][4]) < 0:
                            qs3[2*i+k][5] = -qs3[2*i+k][5] + 2*qs3[2*i+k][0]
                        else:
                            qs3[2*i+k][5] = -qs3[2*i+k][5] - 2*qs3[2*i+k][0]
                        #print tmp1 + tmp2
                        #print T[0,0]
                        #print 'yo2'
                while qs3[2*i+k][5] < 0.:
                    qs3[2*i+k][5] += 2.*pi
                while qs3[2*i+k][5] > 2.*pi:
                    qs3[2*i+k][5] -= 2.*pi
        if debug:
            print 'yeh', qs3[i]

        if False:
            print 'wwwwwwwwwwwwwwww', sin(qs3[i][5]+qs3[i][0]), sin(qs3[i][5]-qs3[i][0]), T[0,0]
            print 'qqqqqqqqqqqqqqqq', cos(qs3[i][5]+qs3[i][0]), cos(qs3[i][5]-qs3[i][0]), T[0,1]
            flip_sign_sin, flip_sign_cos, flip_sub_sin, flip_sub_cos = False, False, False, False
            flip_diff = False
            if abs(abs(sin(qs3[i][5]+qs3[i][0])) - abs(T[0,0])) > ZERO_THRESH:
                qs3[i][5] -= 2*qs3[i][0]
                print 'a'
            print 'wwwwwwwwwwwwwwww', sin(qs3[i][5]+qs3[i][0]), sin(qs3[i][5]-qs3[i][0]), T[0,0]

            if abs(sin(qs3[i][5]+qs3[i][0]) - T[0,0]) > ZERO_THRESH:
                flip_sign_sin = True
            if abs(cos(qs3[i][5]+qs3[i][0]) - T[0,1]) > ZERO_THRESH:
                flip_sign_cos = True
            if flip_sign_sin:
                if flip_sign_cos:
                    qs3[i][5] += pi
                    print 'b'
                else:
                    qs3[i][5] = -qs3[i][5] 
                    #qs3[i][5] = -qs3[i][5] - 2*qs3[i][0]
                    qs3[i][0] = -qs3[i][0]
                    print 'c'
            elif flip_sign_cos:
                qs3[i][5] = pi -qs3[i][5]
                #qs3[i][5] = pi -qs3[i][5] - 2*qs3[i][0]
                qs3[i][0] = -qs3[i][0]
                print 'd'
            print 'e'

            print '3333333333333333', sin(qs3[i][5]+qs3[i][0]), sin(qs3[i][5]-qs3[i][0]), T[0,0]
            print '4444444444444444', cos(qs3[i][5]+qs3[i][0]), cos(qs3[i][5]-qs3[i][0]), T[0,1]
            #qs3[i][0] -= pi
            #qs3[i][0] -= 2*acos
        #if T[0,1] >= 0.:
        #    if -T[2,2] >= 0.:
        #        qs3[i][5] -= pi
        #qs3[i*4+2][5] = 2.*pi - acos
        #qs3[i*4+3][5] = -2.*pi + acos
    print '------- Possibs ----------'
    print np.array(qs3)
    print '--------------------------'
    qs4 = []
    for i in range(len(qs3)):
        c1, s1 = cos(qs3[i][0]), sin(qs3[i][0])
        c5, s5 = cos(qs3[i][4]), sin(qs3[i][4])
        c6, s6 = cos(qs3[i][5]), sin(qs3[i][5])
        x04x = -s5*(T[0,2]*c1 + T[1,2]*s1) - c5*(s6*(T[0,1]*c1 + T[1,1]*s1) - c6*(T[0,0]*c1 + T[1,0]*s1))
        x04y = c5*(T[2,0]*c6 - T[2,1]*s6) - T[2,2]*s5
        p04x = d[4]*(s6*(T[0,0]*c1 + T[1,0]*s1) + c6*(T[0,1]*c1 + T[1,1]*s1)) - d[5]*(T[0,2]*c1 + T[1,2]*s1) + T[0,3]*c1 + T[1,3]*s1
        p04y = T[2,3] - d[0] - d[5]*T[2,2] + d[4]*(T[2,1]*c6 + T[2,0]*s6)
        #_, Ts = forward_kin(qs3[i], a, d, l)
        #T14 = inv_mat(Ts[0]) * T * inv_mat(Ts[5]) * inv_mat(Ts[4])
        #qs_rrr = inverse_rrr(T14, a[1:4], d[1:4])
        if debug:
            print 'lllh', p04x, p04y, x04x, x04y
            print 'kk', c1, s1, c5, s5, c6, s6
        qs_rrr = inverse_rrr(p04x, p04y, x04x, x04y, a[1:4], d[1:4])
        for j in range(len(qs_rrr)):
            qsol = [qs3[i][0], qs_rrr[j][0], qs_rrr[j][1], qs_rrr[j][2], qs3[i][4], qs3[i][5]]
            if abs(-sin(qsol[1] + qsol[2] + qsol[3])*sin(qsol[4]) - T[2,2]) < ZERO_THRESH:
                qs4.append(qsol)
            #Tsol, _ = forward_kin(qsol, a, d, l)
            #print 'yo', qsol
            #print Tsol**-1 * T06
    if False:
        qs4 = np.array(qs4)[np.lexsort(np.mat(qs4).T)[0]]
        unique_sols = []
        qlast = np.array([-999.]*6)
        for i in range(np.size(qs4,0)):
            if np.sum(abs(qlast - qs4[i])) > ZERO_THRESH:
                unique_sols.append(qs4[i])
                qlast = qs4[i]
        return unique_sols
    else:
        return qs4


def inverse_kin(T06, a, d, l, q6_des, debug=False):
    qs1 = []
    T = inv_mat(Tb0) * T06 * inv_mat(T6e)
    A = d[5]*T[1,2] - T[1,3]
    B = d[5]*T[0,2] - T[0,3]
    R = A*A + B*B
    if abs(A) < ZERO_THRESH:
        if abs(abs(d[3]) - abs(B)) < ZERO_THRESH:
            div = -np.sign(d[3])*np.sign(B)
        else:
            div = -d[3]/B
        asin = arcsin(div)
        if asin < 0.:
            qs1.append(asin + 2.*pi)
        else:
            qs1.append(asin)
        qs1.append(pi - asin)
    elif abs(B) < ZERO_THRESH:
        #print 'RRRRR', d[3], A
        if abs(abs(d[3]) - abs(A)) < ZERO_THRESH:
            div = np.sign(d[3])*np.sign(A)
        else:
            div = d[3]/A
        acos = arccos(div)
        qs1.append(acos)
        qs1.append(2.*pi - acos)
    elif d[3]*d[3] > R:
        print '1: Impossible solution'
        return []
    else:
        acos = arccos(d[3] / sqrt(R)) 
        atan = arctan2(-B, A)
        pos = acos + atan
        neg = -acos + atan
        if pos >= 0.:
            qs1.append(pos)
        else:
            qs1.append(2.*pi + pos)
        if neg >= 0.:
            qs1.append(neg)
        else:
            qs1.append(2.*pi + neg)
    qs2 = []
    for q1 in qs1:
        numer = (T[0,3]*sin(q1) - T[1,3]*cos(q1)-d[3])
        if abs(abs(numer) - abs(d[5])) < ZERO_THRESH:
            div = np.sign(numer) * np.sign(d[5])
        else:
            div = numer / d[5]
        acos = arccos(div)
        qs2.append([q1, acos])
        qs2.append([q1, 2.*pi-acos])

    qs3 = []
    for q1, q5 in qs2:
        s5 = sin(q5)
        if abs(s5) < ZERO_THRESH:
            qs3.append([q1, q5, q6_des])
        #elif abs(abs(s5) - 1.) < ZERO_THRESH:
        #    acos = arcsin(T[1,0]) + q1
        #    if acos < 0.:
        #        acos = np.mod(2.*pi - acos, pi/2)
        #    for j in range(4):
        #        qs3.append([q1, q5, acos + j*pi/2])
        #        qs3.append([q1, q5, -acos + (j+1)*pi/2])
        else:
            atan = arctan2(np.sign(s5)*-(T[0,1]*sin(q1) - T[1,1]*cos(q1)), 
                           np.sign(s5)*(T[0,0]*sin(q1) - T[1,0]*cos(q1)))
            if atan < 0.:
                atan += 2.*pi
            qs3.append([q1, q5, atan])
            #numer = (T[0,0]*sin(q1) - T[1,0]*cos(q1))
            #if abs(abs(numer) - abs(s5)) < ZERO_THRESH:
            #    div = np.sign(numer) * np.sign(s5)
            #else:
            #    div = numer / s5
            #acos = arccos(div)
            #if acos < 0.:
            #    acos = 2.*pi+acos
            #qs3.append([q1, q5, 2.*pi-acos])
            #qs3.append([q1, q5, acos])
    #print '------- Possibs ----------'
    #print np.array(qs3)
    #print '--------------------------'
    qs4 = []
    for i, (q1, q5, q6) in enumerate(qs3):
        c1, s1 = cos(q1), sin(q1)
        c5, s5 = cos(q5), sin(q5)
        c6, s6 = cos(q6), sin(q6)
        x04x = -s5*(T[0,2]*c1 + T[1,2]*s1) - c5*(s6*(T[0,1]*c1 + T[1,1]*s1) - c6*(T[0,0]*c1 + T[1,0]*s1))
        x04y = c5*(T[2,0]*c6 - T[2,1]*s6) - T[2,2]*s5
        p04x = d[4]*(s6*(T[0,0]*c1 + T[1,0]*s1) + c6*(T[0,1]*c1 + T[1,1]*s1)) - d[5]*(T[0,2]*c1 + T[1,2]*s1) + T[0,3]*c1 + T[1,3]*s1
        p04y = T[2,3] - d[0] - d[5]*T[2,2] + d[4]*(T[2,1]*c6 + T[2,0]*s6)
        qs_rrr = inverse_rrr(p04x, p04y, x04x, x04y, a[1:4], d[1:4])
        #print 'lllh', p04x, p04y, x04x, x04y
        #if len(qs_rrr) == 0:
        #    print 'No RRR', i
        for j in range(len(qs_rrr)):
            qsol = [q1, qs_rrr[j][0], qs_rrr[j][1], qs_rrr[j][2], q5, q6]
            x = forward_kin_fast(qsol, a, d, l)
            diff = inv_mat(x) * T06
            if debug:
                print qsol
                print x
                print T
                print diff
            qs4.append(qsol)
            #comp_mat = abs(diff - np.eye(4)) < 1e-3
            #if np.all(comp_mat):
            #    qs4.append(qsol)
            ##else:
            ##    print 'Bad sol', i
            #else:
            #    print comp_mat
            #    print diff
            #    print x
            #    print T06
            #    ang, _, _ = mat_to_ang_axis_point(diff)
            #    dist = np.linalg.norm(diff[:3,3])
            #    print ang, dist
            #    #print inv_mat(Ts[0]) * x
            #    #print inv_mat(Ts[0]) * T
            #    sum_comp_mat += 1-comp_mat
            #    num += 1
    #print '------- Possibs2 ---------'
    #print np.array(qs4)
    #print '--------------------------'
    return qs4

def main():
    from ur_cart_move import RAVEKinematics
    np.set_printoptions(precision=3)
    if False:
        d1, a2, a3, d4, d5, d6 = [0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922]
        a = [0, a2, a3, 0, 0, 0]
        d = [d1, 0, 0, d4, d5, d6]
        l = [pi/2, 0, 0, pi/2, -pi/2, 0]
        kin = RAVEKinematics()
        rospy.init_node("test_ur_ik")
        start_time = rospy.get_time()
        n = 0
        while not rospy.is_shutdown():
            q = (np.random.rand(6)-.5)*4*pi
            #Tsol = kin.forward(q)
            T = forward_kin_fast(q, a, d, l)
            #T = forward_kin(q, a, d, l)
            n += 1
            if False:
                diff = inv_mat(Tsol) * T
                #print '-'*40
                #print Tsol
                #print T
                ang, _, _ = mat_to_ang_axis_point(diff)
                dist = np.linalg.norm(diff[:3,3])
                print ang, dist
        time_diff = rospy.get_time() - start_time
        print time_diff, n, n/time_diff

    if True:
        #q = [ 4.07545758,  5.71643082, -4.57552159, -2.79061482, -3.17069678, 1.42865389]
        d1, a2, a3, d4, d5, d6 = [0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922]
        a = [0, a2, a3, 0, 0, 0]
        d = [d1, 0, 0, d4, d5, d6]
        l = [pi/2, 0, 0, pi/2, -pi/2, 0]
        kin = RAVEKinematics()
        rospy.init_node("test_ur_ik")
        start_time = rospy.get_time()
        n = 0
        sum_comp_mat = np.mat(np.zeros((4,4)))
        num = np.array([0.])
        while not rospy.is_shutdown():
            q = (np.random.rand(6)-.5)*4*pi
            x1 = kin.forward(q)
            pos, euler = PoseConv.to_pos_euler(x1)
            m = np.random.randint(-4,5)
            euler = [euler[0], m*np.pi/2 + 0., euler[2]]
            #euler = [euler[0], 0.*np.pi/2 + m*np.pi, euler[2]]
            T = PoseConv.to_homo_mat(pos, euler)
            #q[4] = 0.
            T = kin.forward(q)
            #sols = inverse_kin(T,a,d,l, q[5], sum_comp_mat, num)
            sols = inverse_kin(T,a,d,l, q[5])
            print m, len(sols)
            if len(sols) == 0:
                print 'wuh', T
                print 'huh', q
                print 'kuk', euler
                #sols = inverse_kin(T,a,d,l, q[5], sum_comp_mat, num, True)
                sols = inverse_kin(T,a,d,l, q[5], True)
                
            if False:
                for qsol in sols:
                    #Tsol, _ = forward_kin(qsol, a, d, l)
                    Tsol = kin.forward(qsol)
                    #print qsol
                    #print q
                    #print Tsol
                    #print T
                    diff = inv_mat(Tsol) * T
                    if True:
                        ang, _, _ = mat_to_ang_axis_point(diff)
                        dist = np.linalg.norm(diff[:3,3])
                        #print ang, dist
                        if abs(dist) > 1e-5 or abs(ang) > 1e-5:
                            print 'BAD'
                        else:
                            pass
                            #print 'GOOD'
            n += 1
        time_diff = rospy.get_time() - start_time
        print time_diff, n, n/time_diff
        print sum_comp_mat, num

    if False:
        #q = [ 4.07545758,  5.71643082, -4.57552159, -2.79061482, -3.17069678, 1.42865389]
        d1, a2, a3, d4, d5, d6 = [0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922]
        a = [0, a2, a3, 0, 0, 0]
        d = [d1, 0, 0, d4, d5, d6]
        l = [pi/2, 0, 0, pi/2, -pi/2, 0]
        kin = RAVEKinematics()
        rospy.init_node("test_ur_ik")
        start_time = rospy.get_time()
        n = 0
        while not rospy.is_shutdown():
            q = (np.random.rand(6)-.5)*4*pi
            T = kin.forward(q)
            sols = inverse_kin(T,a,d,l)
            #print len(sols)
            if False:
                print len(sols)
                for qsol in sols:
                    #Tsol, _ = forward_kin(qsol, a, d, l)
                    Tsol = kin.forward(qsol)
                    diff = Tsol**-1 * T
                    ang, _, _ = mat_to_ang_axis_point(diff)
                    dist = np.linalg.norm(diff[:3,3])
                    #print ang, dist
                    if abs(dist) > 1e-8 or abs(ang) > 1e-8:
                        print 'BAD'
            n += 1
        time_diff = rospy.get_time() - start_time
        print time_diff, n, n/time_diff
        
    if False:
        #q = [ 4.07545758,  5.71643082, -4.57552159, -2.79061482, -3.17069678, 1.42865389]
        q = (np.random.rand(6)-.5)*4*pi
        d1, a2, a3, d4, d5, d6 = [0.1273, -0.612, -0.5723, 0.163941, 0.1157, 0.0922]
        a = [0, a2, a3, 0, 0, 0]
        d = [d1, 0, 0, d4, d5, d6]
        l = [pi/2, 0, 0, pi/2, -pi/2, 0]
        kin = RAVEKinematics()
        T = kin.forward(q)
        print T
        print forward_kin(q,a,d,l)
        print q
        sols = inverse_kin(T,a,d,l)
        for qsol in sols:
            Tsol, _ = forward_kin(qsol, a, d, l)
            diff = Tsol**-1 * T
            ang, _, _ = mat_to_ang_axis_point(diff)
            dist = np.linalg.norm(diff[:3,3])
            if False:
                if abs(dist) > 1e-6:
                    print '-'*80
                else:
                    print '+'*80
                print 'T', T
                print 'qsol', qsol
                print 'q234', np.sum(qsol[1:4])
                print 'q5', qsol[4]
                print '-sin(q2 + q3 + q4)*sin(q5)', -sin(qsol[1] + qsol[2] + qsol[3])*sin(qsol[4])
                print 'z3', T[2,0]
                if abs(dist) > 1e-6:
                    print '-'*80
                else:
                    print '+'*80
            print ang, dist
        print np.sort(sols,0)
        print len(sols)
        #unique_sols = np.array(sols)[np.where(np.hstack(([True], np.sum(np.diff(np.sort(sols,0), 1, 0),1) > 0.000)))[0]]
        #print unique_sols
        #print len(unique_sols)
        #print len(np.hstack(([True], np.sum(np.diff(np.sort(sols,0), 1, 0),1) > 0.000)))
        for qsol in sols:
            #Tsol, _ = forward_kin(qsol, a, d, l)
            Tsol = kin.forward(qsol)
            diff = Tsol**-1 * T
            ang, _, _ = mat_to_ang_axis_point(diff)
            dist = np.linalg.norm(diff[:3,3])
            print ang, dist
            if abs(dist) > 1e-8:
                print 'BAD'
        
        kin.robot.SetDOFValues(np.array([0.]*6))
        rave_sols = kin.manip.FindIKSolutions(T.A, kin.ik_options)
        rave_list = []
        for qsol in rave_sols:
            rave_list.append(np.array(qsol))
            #Tsol, _ = forward_kin(qsol, a, d, l)
            Tsol = kin.forward(qsol)
            diff = Tsol**-1 * T
            ang, _, _ = mat_to_ang_axis_point(diff)
            dist = np.linalg.norm(diff[:3,3])
            print ang, dist
            if abs(dist) > 1e-8:
                print 'BAD'
        print np.sort(rave_list,0)
            #print diff
            #print q
            #print qsol
            #print '-'*80

    if False:
        q = (np.random.rand(3)-0.5)*4.*np.pi
        T04 = forward_rrr(q)
        print T04
        qs = inverse_rrr(T04)
        print qs
        print T04**-1 * forward_rrr(qs[0])
        print T04**-1 * forward_rrr(qs[1])

if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', 'prof')
    #main()
