import roslib
from pushing_dynamics_models import *
import dynamics_learning

def train_and_save_svr_dynamics(train_file_base_name, svr_output_path,
                                delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                kernel_type='LINEAR'):
    # Read data from disk
    print 'Reading training file:', train_file_base_name
    (train_X, train_Y) = dynamics_learning.read_dynamics_learning_example_files(train_file_base_name)
    # print 'train_X:\n',train_X
    # print 'train_Y:\n',train_Y
    print 'len(train_X)',len(train_X)
    print 'len(train_Y)',len(train_Y)
    # Build model class
    svr_dynamics = SVRPushDynamics(delta_t, n, m, epsilons=epsilons,
                                   feature_names = feature_names,
                                   target_names = target_names,
                                   xtra_names = xtra_names,
                                   kernel_type = kernel_type)
    # Train and save model
    svr_dynamics.learn_model(train_X, train_Y)
    print 'Saving model to:', svr_output_path
    svr_dynamics.save_models(svr_output_path)
    return svr_dynamics

def setup_leave_one_out_and_single_class_models():
    all_classes = ['bear', 'food_box',  'phone', 'large_brush0', 'soap_box',
                   'camcorder', 'glad', 'salt', 'batteries', 'mug',
                   'shampoo', 'bowl', 'large_vitamins', 'plate', 'water_bottle']

    # HACK: Test first
    train_classes = ['bear', 'food_box']

    # SVR options
    kernel_type = 'LINEAR'
    delta_t = 1./9.
    n = 6
    m = 3
    target_names = [dynamics_learning._DELTA_OBJ_X_OBJ,
                    dynamics_learning._DELTA_OBJ_Y_OBJ,
                    dynamics_learning._DELTA_OBJ_THETA_WORLD,
                    dynamics_learning._DELTA_EE_X_OBJ,
                    dynamics_learning._DELTA_EE_Y_OBJ,
                    dynamics_learning._DELTA_EE_PHI_WORLD]
    feature_names = [dynamics_learning._EE_X_OBJ,
                     dynamics_learning._EE_Y_OBJ,
                     dynamics_learning._EE_PHI_OBJ,
                     dynamics_learning._U_X_OBJ,
                     dynamics_learning._U_Y_OBJ,
                     dynamics_learning._U_PHI_WORLD]
    xtra_names = []
    epsilons = []
    for i in xrange(len(target_names)):
        epsilons.append(1e-4)

    # Train and save models
    base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/'
    example_in_dir = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/'
    base_example_hold_out_dir_name = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/single_obj/'
    base_example_single_obj_dir_name = base_example_hold_out_dir_name

    for obj_class in train_classes:
        hold_out_classes = all_classes[:]
        hold_out_classes.remove(obj_class)

        print 'Creating holdout files for object:', obj_class
        dynamics_learning.create_train_and_validate_obj_class_splits(example_in_dir,
                                                                     base_example_hold_out_dir_name,
                                                                     hold_out_classes)
        # Train model with obj class as only left out class
        hold_out_str = ''
        for hold_out_class in hold_out_classes:
            hold_out_str += '_' + hold_out_class

        hold_out_train_file_base_name = base_example_hold_out_dir_name + 'objs' + hold_out_str
        svr_hold_out_output_path = base_svr_path + 'hold_out_' + obj_class

        train_and_save_svr_dynamics(hold_out_train_file_base_name, svr_hold_out_output_path,
                                    delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                    kernel_type='LINEAR')

        # Train model with obj class as only training class
        single_obj_train_file_base_name = base_example_single_obj_dir_name + 'objs_' + obj_class
        svr_single_obj_output_path = base_svr_path + 'single_obj_' + obj_class

        train_and_save_svr_dynamics(single_obj_train_file_base_name, svr_single_obj_output_path,
                                    delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                    kernel_type='LINEAR')

def train_shape_clusters():
    # TODO: Get object classes based on shape similarity
    pass

def train_random_object_clusters():
    # TODO: Get object classes based on shape similarity
    pass
