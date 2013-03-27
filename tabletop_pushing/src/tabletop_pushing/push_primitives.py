# To add a new controller:
#     need to put in correct switch in tracker_feedback_push() in positon_feedback_push_node.py
#     need to add admissable primitives to BEHAVIOR_PRIMITIVES
#     need to add admissable proxies to PERCEPTUAL_PROXIES
# To add a new primitive:
#     need to put in perform_push() in tabletop_executive.py
#     need to add which controllers can use it in BEHAVIOR_PRIMITIVES (maybe add to  another PUSH_PRIMITIVES, TOOL_PRIMITIVES, etc.)
#     need to add a precondiiton method for it in PRECONDITION_METHODS
# To add a new proxy: TODO: Check this by adding a new proxy
#     need to put in computeState() in tabletop_pushing_perception_node.cpp
#     need to add which controllers can use it in PERCEPTUAL_PROXIES (maybe add to  another CENTROID_PROXIES, POSE_PROXIES, etc.)
#     need to add a string constant for the name in tabletop_pushing_perception_node.cpp

ROBOT_ARMS = ['r', 'l']
ROBOT_ARMS = ['r']
# ROBOT_ARMS = ['l']
CENTROID_CONTROLLER ='centroid_controller'
TOOL_CENTROID_CONTROLLER ='tool_centroid_controller'
SPIN_COMPENSATION = 'spin_compensation'
SPIN_TO_HEADING = 'spin_to_heading'
STRAIGHT_LINE_CONTROLLER = 'straight_line_controller'
DIRECT_GOAL_CONTROLLER = 'direct_goal_controller'
DIRECT_GOAL_GRIPPER_CONTROLLER = 'direct_goal_gripper_controller'
CONTROLLERS = [CENTROID_CONTROLLER, SPIN_COMPENSATION, DIRECT_GOAL_CONTROLLER]
RBF_CONTROLLER_PREFIX = 'RBF_'
AFFINE_CONTROLLER_PREFIX = 'AFFINE_'
RBF = 'RBF_push_learn_mgp_35251.84'
CONTROLLERS = [CENTROID_CONTROLLER]
# CONTROLLERS = [SPIN_COMPENSATION]
# CONTROLLERS = [DIRECT_GOAL_CONTROLLER]
# CONTROLLERS = [STRAIGHT_LINE_CONTROLLER]
# CONTROLLERS = [TOOL_CENTROID_CONTROLLER]

# CONTROLLERS = [RBF]
GRIPPER_PUSH = 'gripper_push'
GRIPPER_SWEEP = 'gripper_sweep'
TOOL_SWEEP = 'tool_sweep'
TOOL_POKE = 'tool_poke'
OVERHEAD_PUSH = 'overhead_push'
OPEN_OVERHEAD_PUSH = 'open_overhead_push'
PINCHER_PUSH = 'pincher_push'
GRIPPER_PULL = 'gripper_pull'
PUSH_PRIMITIVES = [OVERHEAD_PUSH, GRIPPER_PUSH, GRIPPER_SWEEP, OPEN_OVERHEAD_PUSH, PINCHER_PUSH]
PUSH_PRIMITIVES = [OVERHEAD_PUSH]
PUSH_PRIMITIVES = [GRIPPER_PUSH]
TOOL_PRIMITIVES = [TOOL_SWEEP]
BEHAVIOR_PRIMITIVES = {CENTROID_CONTROLLER:PUSH_PRIMITIVES, SPIN_COMPENSATION:PUSH_PRIMITIVES,
                       STRAIGHT_LINE_CONTROLLER:PUSH_PRIMITIVES,
                       DIRECT_GOAL_CONTROLLER:[GRIPPER_PULL],
                       DIRECT_GOAL_GRIPPER_CONTROLLER:[GRIPPER_PULL],
                       TOOL_CENTROID_CONTROLLER:TOOL_PRIMITIVES,
                       RBF:PUSH_PRIMITIVES}

ELLIPSE_PROXY = 'ellipse'
CENTROID_PROXY = 'centroid'
SPHERE_PROXY = 'sphere'
CYLINDER_PROXY = 'cylinder'
BOUNDING_BOX_XY_PROXY = 'bounding_box_xy'

HACK_TOOL_PROXY = 'hack'
EE_TOOL_PROXY = 'end_effector_tool'

CENTROID_PROXIES = [CENTROID_PROXY, SPHERE_PROXY, BOUNDING_BOX_XY_PROXY]
CENTROID_PROXIES = [ELLIPSE_PROXY]
POSE_PROXIES = [ELLIPSE_PROXY, BOUNDING_BOX_XY_PROXY]

PERCEPTUAL_PROXIES = {CENTROID_CONTROLLER:CENTROID_PROXIES,
                      SPIN_COMPENSATION:POSE_PROXIES,
                      STRAIGHT_LINE_CONTROLLER:CENTROID_PROXIES,
                      DIRECT_GOAL_CONTROLLER:CENTROID_PROXIES,
                      DIRECT_GOAL_GRIPPER_CONTROLLER:CENTROID_PROXIES,
                      TOOL_CENTROID_CONTROLLER:CENTROID_PROXIES,
                      RBF:POSE_PROXIES}

CENTROID_PUSH_PRECONDITION = 'centroid_push'
CENTROID_PULL_PRECONDITION = 'centroid_pull'
TOOL_SWEEP_PRECONDITION = 'tool_sweep'
PRECONDITION_METHODS = {GRIPPER_PULL:CENTROID_PULL_PRECONDITION,
                        OVERHEAD_PUSH:CENTROID_PUSH_PRECONDITION,
                        OPEN_OVERHEAD_PUSH:CENTROID_PUSH_PRECONDITION,
                        GRIPPER_PUSH:CENTROID_PUSH_PRECONDITION,
                        GRIPPER_SWEEP:CENTROID_PUSH_PRECONDITION,
                        PINCHER_PUSH:CENTROID_PUSH_PRECONDITION,
                        TOOL_SWEEP:TOOL_SWEEP_PRECONDITION}
