import roslib
from pushing_dynamics_models import *
import dynamics_learning
import numpy as np

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

def analyze_pred_vs_gt(Y_hat, Y_gt):
    Y_errors = []
    error_means = []
    error_sds = []
    for Y_hat_i, Y_gt_i in zip(Y_hat, Y_gt):
        error_i = np.abs(np.array(Y_hat_i) - np.array(Y_gt_i))
        Y_errors.append(error_i)
        error_means.append(np.mean(error_i))
        error_sds.append(np.sqrt(np.var(error_i)))
    return (error_means, error_sds, Y_errors)

def test_svr_offline(model_param_file_name, test_data_example_base_name):
    svr_dynamics = SVRPushDynamics(param_file_name = model_param_file_name)
    (test_X, test_Y) = dynamics_learning.read_dynamics_learning_example_files(test_data_example_base_name)
    # Run batch testing
    (Y_hat, Y_gt, _) = svr_dynamics.test_batch_data(test_X, test_Y)
    return (Y_hat, Y_gt)

def build_results_table(error_means_all, error_sds_all, error_diffs_all,
                        target_names, test_obj_names, hold_out_obj_name,
                        table_out_path=''):
    # Group errors by output dimension
    overall_errors = []
    for i in xrange(len(target_names)):
        overall_errors.append(np.array([]))
        for error_diffs in error_diffs_all:
            overall_errors[i] = np.concatenate((overall_errors[i], error_diffs[i]))

    # Get mean and stand dev for each output dimension
    overall_means = []
    overall_sds = []
    for overall_error in overall_errors:
        overall_means.append(np.mean(overall_error))
        overall_sds.append(np.sqrt(np.var(overall_error)))

    title_line = '|Hold out object: ' + hold_out_obj_name +'\t|'
    header_line = '|\t|'
    for target_name in target_names:
        title_line += '\t|'
        header_line += target_name + '\t|'

    print title_line
    print header_line
    title_line += '\n'
    header_line += '\n'

    table_mean_lines = []
    table_sds_lines = []

    # Output to terminal
    for error_means, error_sds, test_obj_name in zip(error_means_all, error_sds_all, test_obj_names):
        mean_str = ''
        sds_str = ''
        for mean in error_means:
            mean_str += str(mean) + '\t|'
        for sd in error_sds:
            sds_str += str(sd) + '\t|'
        table_mean_line = '|' + test_obj_name + '\t|' + mean_str
        table_sds_line = '|' + test_obj_name + '\t|' + sds_str
        print table_mean_line
        print table_sds_line
        table_mean_line += '\n'
        table_sds_line += '\n'
        table_mean_lines.append(table_mean_line)
        table_sds_lines.append(table_sds_line)

    overall_mean_line = '|Overall\t|'
    overall_sds_line = '|Overall\t|'
    for mean in overall_means:
        overall_mean_line += str(mean) + '\t|'
    for sd in overall_sds:
        overall_sds_line += str(sd) + '\t|'
    print overall_mean_line
    print overall_sds_line
    overall_mean_line += '\n'
    overall_sds_line += '\n'

    # Output to disk
    if len(table_out_path) > 0:
        mean_out_name = table_out_path + hold_out_obj_name + '-means.txt'
        mean_out_file = file(mean_out_name, 'w')
        mean_out_file.write(title_line)
        mean_out_file.write(header_line)
        mean_out_file.writelines(table_mean_lines)
        mean_out_file.write(overall_mean_line)
        mean_out_file.close()

        sds_out_name = table_out_path + hold_out_obj_name + '-sds.txt'
        sds_out_file = file(sds_out_name, 'w')
        sds_out_file.write(title_line)
        sds_out_file.write(header_line)
        sds_out_file.writelines(table_sds_lines)
        sds_out_file.write(overall_sds_line)
        sds_out_file.close()

def compare_obj_class_results():
    all_classes = ['bear', 'food_box',  'phone', 'large_brush0', 'soap_box',
                   'camcorder', 'glad', 'salt', 'batteries', 'mug',
                   'shampoo', 'bowl', 'large_vitamins', 'plate', 'water_bottle']
    target_names = [dynamics_learning._DELTA_OBJ_X_OBJ,
                    dynamics_learning._DELTA_OBJ_Y_OBJ,
                    dynamics_learning._DELTA_OBJ_THETA_WORLD,
                    dynamics_learning._DELTA_EE_X_OBJ,
                    dynamics_learning._DELTA_EE_Y_OBJ,
                    dynamics_learning._DELTA_EE_PHI_WORLD]

    hold_out_classes = all_classes[:]

    # Go through all objects
    base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/'
    base_example_dir_name = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/single_obj/'
    for hold_out_class in hold_out_classes:

        test_classes = all_classes[:]
        test_classes.remove(hold_out_class)
        model_param_file_name = base_svr_path + 'single_obj_' + hold_out_class + '_params.txt'
        # TODO: Setup an output path
        table_out_path = '/u/thermans/sandbox/'
        # Test against object files for each of the held out objects independently
        Y_hat_all = []
        Y_gt_all = []
        error_means_all = []
        error_sds_all = []
        error_diffs_all = []
        for test_obj in test_classes:
            print '\nTesting for object', test_obj, 'with model trained on', hold_out_class
            test_obj_example_base_name = base_example_dir_name + 'objs_' + test_obj
            (Y_hat, Y_gt) = test_svr_offline(model_param_file_name, test_obj_example_base_name)
            # Analyze output
            (error_means, error_sds, error_diffs) = analyze_pred_vs_gt(Y_hat, Y_gt)
            Y_hat_all.append(Y_hat)
            Y_gt_all.append(Y_hat)
            error_means_all.append(error_means)
            error_sds_all.append(error_sds)
            error_diffs_all.append(error_diffs)
        # TODO: Build table for data and save table to disk
        build_results_table(error_means_all, error_sds_all, error_diffs_all,
                           target_names, test_classes, hold_out_class,
                           table_out_path)

def setup_leave_one_out_and_single_class_models():
    all_classes = ['bear', 'food_box',  'phone', 'large_brush0', 'soap_box',
                   'camcorder', 'glad', 'salt', 'batteries', 'mug',
                   'shampoo', 'bowl', 'large_vitamins', 'plate', 'water_bottle']

    # HACK: Test first
    train_classes = all_classes[2:]

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
