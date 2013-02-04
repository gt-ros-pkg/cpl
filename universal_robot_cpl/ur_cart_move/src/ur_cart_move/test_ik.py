from openravepy import *
import numpy as np, time
import roslib
roslib.load_manifest('pykdl_utils')
from pykdl_utils.kdl_kinematics import create_kdl_kin
env = Environment() # create the environment
#env.SetViewer('qtcoin') # start the viewer
env.Load('ur10_robot.dae') # load a scene
robot = env.GetRobots()[0] # get the first robot
kdl_kin = create_kdl_kin('base_link', 'ee_link')

manip = robot.SetActiveManipulator('arm') # set the manipulator to leftarm
ikmodel = databases.inversekinematics.InverseKinematicsModel(robot,iktype=IkParameterization.Type.Transform6D)
if ikmodel.load():
    print 'model loaded'
else:
    ikmodel.autogenerate()

#Tgoal = np.mat(manip.GetEndEffectorTransform())
#print Tgoal, kdl_kin.forward(robot.GetDOFValues())
#print kdl_kin.inverse(Tgoal)
#Tgoal[0,3] -= 0.3
#print Tgoal

options = (IkFilterOptions.IgnoreSelfCollisions |
           IkFilterOptions.IgnoreCustomFilters |
           IkFilterOptions.IgnoreEndEffectorCollisions |
           IkFilterOptions.IgnoreJointLimits)

def test_iks():
    n = 1250
    q_rands, x_news, x_twists = [], [], []
    for i in range(n):
        q_rand = np.random.rand(6) * np.pi *2 - np.pi
        x_init = kdl_kin.forward(q_rand)
        x_rand = np.mat(np.eye(4))
        x_twist = np.mat(0.03*(np.random.rand(6)-0.5)/0.5).T
        x_twist[3:,0] = 0.0
        x_rand[:3,3] = x_twist[:3,0]
        x_new = x_rand*x_init
        q_rands.append(q_rand)
        x_news.append(x_new)
        x_twists.append(x_twist)
    sols_kdl, sols_rave = [], []
    jacs_kdl, jacs_rave = [], []

    print 'kdl jacobians'
    start_time = time.time()
    for i in range(n):
        jac = kdl_kin.jacobian(q_rands[i])
        jacs_kdl.append(np.linalg.inv(jac))
    time_elapsed = (time.time() - start_time)
    print time_elapsed, n, time_elapsed/n, n/time_elapsed

    print 'kdl ik'
    nones = 0
    start_time = time.time()
    for i in range(n):
        sol = kdl_kin.inverse(x_news[i], q_guess=q_rands[i])
        if sol is None:
            nones += 1
        sols_kdl.append(sol)
    time_elapsed = (time.time() - start_time)
    print 'nones', nones
    print time_elapsed, n, time_elapsed/n, n/time_elapsed

    print 'kdl ik + jac'
    nones = 0
    start_time = time.time()
    for i in range(n):
        sol = kdl_kin.inverse(x_news[i], q_guess=q_rands[i]+0.4*(jacs_kdl[i]*x_twists[i]).T.A[0])
        if sol is None:
            nones += 1
        sols_kdl.append(sol)
    time_elapsed = (time.time() - start_time)
    print 'nones', nones
    print time_elapsed, n, time_elapsed/n, n/time_elapsed

    print 'rave ik'
    nones = 0
    start_time = time.time()
    for i in range(n):
        #with robot: # save robot state
        #    robot.SetDOFValues(q_rands[i],range(6)) # set the current solution
        #    env.UpdatePublishedBodies() # allow viewer to update new robot
        with env: # lock environment
            if manip.FindIKSolution(x_news[i].A, options) is None:
                nones += 1
    time_elapsed = (time.time() - start_time)
    print 'nones', nones
    print time_elapsed, n, time_elapsed/n, n/time_elapsed

#test_iks()

class RAVEKinematics(object):
    def __init__(self, robot_file, load_ik=True):
        self.env = Environment()
        self.env.Load('ur10_robot.dae')
        self.robot = self.env.GetRobots()[0] # get the first robot
        #self.manip = self.robot.SetActiveManipulator('arm')
        self.manip = self.robot.GetManipulators()[0]
        self.ikmodel = databases.inversekinematics.InverseKinematicsModel(
                             self.robot,iktype=IkParameterization.Type.Transform6D)
        self.options = (IkFilterOptions.IgnoreSelfCollisions |
                        IkFilterOptions.IgnoreCustomFilters |
                        IkFilterOptions.IgnoreEndEffectorCollisions |
                        IkFilterOptions.IgnoreJointLimits)
        if load_ik:
            self.load_ik_model()

    def load_ik_model(self):
        if self.ikmodel.load():
            print 'Model loaded'
        else:
            print 'No model, generating...',
            self.ikmodel.autogenerate()
            print 'model generated!'

    def forward(self, q):
        self.robot.SetDOFValues(q)
        return np.mat(self.manip.GetEndEffectorTransform())

    def inverse(self, x, q_guess=None, options=None):
        if options is None:
            options = self.options
        if q_guess is None:
            q_guess = np.zeros(6)
        with self.env: # lock environment
            sols = self.manip.FindIKSolutions(x.A, options)
            return sols[np.argmin(np.sum((sols - np.array(q_guess))**2,1))]


np.random.seed(3)
q_rand = np.random.rand(6)
robot.SetDOFValues(q_rand)

rave_kin = RAVEKinematics('ur10_robot.dae')
print q_rand
print rave_kin.forward(q_rand)
print rave_kin.inverse(rave_kin.forward(q_rand),q_rand)
print rave_kin.forward(rave_kin.inverse(rave_kin.forward(q_rand)))

def min_jerk_traj(d, n):
    return [10.0 * (t/d)**3 - 15.0 * (t/d)**4 + 6 * (t/d)**5 for t in np.linspace(0.,d,n)]


from scipy.interpolate import UnivariateSpline
#def try_lin(d, n):
#    with env: # lock environment
#        x_rand = np.mat(manip.GetEndEffectorTransform())
#        x_cur = x_rand.copy()
#        q_pts = []
#        t_traj = np.linspace(0.,d,n)
#        s_traj = min_jerk_traj(d, n)
#        for s in s_traj:
#            x_cur[0,3] = x_rand[0,3] + 0.3*s
#            q_pt = manip.FindIKSolutions(x_cur.A, options)[0]
#            if q_pt is None:
#                return None
#            q_pts.append(q_pt)
#        q_pts = np.array(q_pts).T
#        splines = [UnivariateSpline(t_traj, q_pts[i]) for i in range(6)]
#        return splines

#start_time = time.time()
#n = 100
#for i in range(n):
#    #print try_lin() is not None
#    try_lin(5., 10)
#time_elapsed = (time.time() - start_time)
#print time_elapsed, n, time_elapsed/n, n/time_elapsed
rave_kin = RAVEKinematics('ur10_robot.dae')
print q_rand
print rave_kin.forward(q_rand)
print rave_kin.inverse(rave_kin.forward(q_rand),q_rand)
print rave_kin.forward(rave_kin.inverse(rave_kin.forward(q_rand)))
def try_lin(d, n):
    rave_kin = RAVEKinematics('ur10_robot.dae')
    x_rand = rave_kin.forward(q_rand)
    x_cur = x_rand.copy()
    q_pts = []
    t_traj = np.linspace(0.,d,n)
    s_traj = min_jerk_traj(d, n)
    q_prev = q_rand
    for s in s_traj:
        x_cur[0,3] = x_rand[0,3] + -0.3*s
        print q_prev
        q_pt = rave_kin.inverse(x_cur, q_prev)
        if q_pt is None:
            return None
        q_pts.append(q_pt)
        q_prev = q_pt
    q_pts = np.array(q_pts).T
    splines = [UnivariateSpline(t_traj, q_pts[i]) for i in range(6)]
    return splines

splines = try_lin(5., 10)
print q_rand
for i in range(6):
    print i+1
    print np.array([splines[i].derivatives(t) for t in np.linspace(0.,5.,30)])
