CENTROID_CONTROLLER ='centroid_controller'
SPIN_COMPENSATION = 'spin_compensation'
SPIN_TO_HEADING = 'spin_to_heading'
DIRECT_GOAL_CONTROLLER = 'direct_goal_controller'
DIRECT_GOAL_GRIPPER_CONTROLLER = 'direct_goal_gripper_controller'
CONTROLLERS = [CENTROID_CONTROLLER, SPIN_COMPENSATION, DIRECT_GOAL_CONTROLLER]
# CONTROLLERS = [SPIN_COMPENSATION]
# CONTROLLERS = [DIRECT_GOAL_CONTROLLER]
# CONTROLLERS = [CENTROID_CONTROLLER]

GRIPPER_PUSH = 'gripper_push'
GRIPPER_SWEEP = 'gripper_sweep'
OVERHEAD_PUSH = 'overhead_push'
GRIPPER_PULL = 'gripper_pull'
PUSH_PRIMITIVES = [OVERHEAD_PUSH, GRIPPER_PUSH, GRIPPER_SWEEP]
# PUSH_PRIMITIVES = [GRIPPER_SWEEP]
ACTION_PRIMITIVES = {CENTROID_CONTROLLER:PUSH_PRIMITIVES, SPIN_COMPENSATION:PUSH_PRIMITIVES,
                     DIRECT_GOAL_CONTROLLER:[GRIPPER_PULL],
                     DIRECT_GOAL_GRIPPER_CONTROLLER:[GRIPPER_PULL]}

ELLIPSE_PROXY = 'ellipse'
CENTROID_PROXY = 'centroid'
SPHERE_PROXY = 'sphere'
CYLINDER_PROXY = 'cylinder'
BOUNDING_BOX_XY_PROXY = 'bounding_box_xy'

CENTROID_PROXIES = [CENTROID_PROXY, SPHERE_PROXY, BOUNDING_BOX_XY_PROXY]
POSE_PROXIES = [ELLIPSE_PROXY,BOUNDING_BOX_XY_PROXY]

PERCEPTUAL_PROXIES = {CENTROID_CONTROLLER:CENTROID_PROXIES,
                      SPIN_COMPENSATION:POSE_PROXIES,
                      DIRECT_GOAL_CONTROLLER:CENTROID_PROXIES,
                      DIRECT_GOAL_GRIPPER_CONTROLLER:CENTROID_PROXIES}

CENTROID_PUSH_PRECONDITION = 'centroid_push'
CENTROID_PULL_PRECONDITION = 'centroid_pull'
PRECONDITION_METHODS = {GRIPPER_PULL:CENTROID_PULL_PRECONDITION,
                        OVERHEAD_PUSH:CENTROID_PUSH_PRECONDITION,
                        GRIPPER_PUSH:CENTROID_PUSH_PRECONDITION,
                        GRIPPER_SWEEP:CENTROID_PUSH_PRECONDITION}
