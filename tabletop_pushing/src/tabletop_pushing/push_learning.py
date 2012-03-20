import roslib; roslib.load_manifest('tabletop_pushing')
from geometry_msgs.msg import Point

_HEADER_LINE = '# c_x c_y c_z theta push_opt arm c_x\' c_y\' c_z\' push_dist'

class PushTrial:
    def __init__(self):
        self.c_x = None
        self.c_x_prime = None
        self.push_angle = None
        self.push_opt = None
        self.arm = None
        self.push_dist = None

    def __str__(self):
        return str((self.c_x, self.push_angle, self.push_opt, self.arm, self.c_x_prime, self.push_dist))

class PushLearningAnalysis:

    def __init__(self):
        self.raw_data = None
        self.io = PushLearningIO()

    def read_in_data(self, file_name):
        self.raw_data = self.io.read_in_data_file(file_name)

    def show_max_vectors(self):
        pass

    def group_data(self):
        for d in self.raw_data():
            # Group by angle
            pass

    def hash_angle(self, theta):
        return theta

    def hash_centroid(self, c):
        # TODO: discretize to .05 meter interval
        # TODO: Put in two closest bins in x and y directions...
        return (c.x, c.y)

    def determine_max_vectors(self, file_name):
        self.read_in_data(file_name)
        # TODO: Get error scores for each push
        # TODO: Group different pushes by push_angle
        # TODO: Group different pushes by start centroid
        # TODO: Choose best push for each (angle, centroid) group

    def compute_push_score(self, push):
        return 0.0

class PushLearningIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, c_x, push_angle, push_opt, arm, c_x_prime, push_dist):
        if self.data_out is None:
            print 'ERROR: Attempting to write to file that has not been opened.'
            return
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z' push_dist
        data_line = str(c_x.x)+' '+str(c_x.y)+' '+str(c_x.z)+' '+\
            str(push_angle)+' '+str(push_opt)+' '+str(arm)+' '+\
            str(c_x_prime.x)+' '+str(c_x_prime.y)+' '+str(c_x_prime.z)+' '+\
            str(push_dist)+'\n'
        self.data_out.write(data_line)

    def parse_line(self, line):
        if line.startswith('#'):
            return None
        l  = line.split()
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z' push_dist
        push = PushTrial()
        push.c_x = Point(float(l[0]), float(l[1]), float(l[2]))
        push.push_angle = float(l[3])
        push.push_opt = int(l[4])
        push.arm = l[5]
        push.c_x_prime = Point(float(l[6]),float(l[7]),float(l[8]))
        push.push_dist = float(l[9])
        return push

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        self.data_out.write(_HEADER_LINE+'\n')

    def close_out_file(self):
        self.data_out.close()
