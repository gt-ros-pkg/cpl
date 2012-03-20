import roslib; roslib.load_manifest('tabletop_pushing')
from geometry_msgs.msg import Point

class PushLearningIO:
    def __init__(self):
        self.data_out = None
        self.data_in = None

    def write_line(self, c_x, push_angle, push_opt, which_arm, c_x_prime):
        if self.data_out is None:
            print 'ERROR: Attempting to write to file that has not been opened.'
            return
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z'
        data_line = str(c_x.x)+' '+str(c_x.y)+' '+str(c_x.z)+' '+\
            str(push_angle)+' '+str(push_opt)+' '+str(which_arm)+' '+\
            str(c_x_prime.x)+' '+str(c_x_prime.y)+' '+str(c_x_prime.z)+'\n'
        self.data_out.write(data_line)

    def parse_line(self, line):
        l  = line.split()
        # c_x c_y c_z theta push_opt arm c_x' c_y' c_z'
        c_x = Point(float(l[0]), float(l[1]), float(l[2]))
        push_angle = float(l[3])
        push_opt = int(l[4])
        which_arm = l[5]
        c_x_prime = Point(float(l[6]),float(l[7]),float(l[8]))
        return (c_x, push_angle, push_opt, which_arm, c_x_prime)

    def read_in_data_file(self, file_name):
        data_in = file(file_name, 'r')
        x = [if not l.startwith('#'): self.parse_line(l) for l in data_in.readlines()]
        data_in.close()
        return filter(None, x)

    def open_out_file(self, file_name):
        self.data_out = file(file_name, 'a')
        # TODO: Write header?

    def close_out_file(self):
        self.data_out.close()
