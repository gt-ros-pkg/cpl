Header header
int8[] bin_slots # bin number at particular bin location, -1 if empty
int8[] reachable_slots # list of slots reachable by human, shouldn't change for experiment
bool robot_is_moving
int8 robot_bin_grasped # -1 if bin is gripper is empty
BinInference inference
