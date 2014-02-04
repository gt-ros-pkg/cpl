import roslib; roslib.load_manifest('tabletop_pushing')
import tf
from pushing_dynamics_models import *
import push_learning
import dynamics_learning
import numpy as np
import os
import subprocess
import random
import util

_ALL_CLASSES = ['bear', 'food_box',  'phone', 'large_brush0', 'soap_box',
               'camcorder', 'glad', 'salt', 'batteries', 'mug',
               'shampoo', 'bowl', 'large_vitamins', 'plate', 'water_bottle']
_TEST_CLASSES = ['bear', 'glad', 'soap_box', 'bowl', 'shampoo', 'large_brush']

_LEARN_SVR_ERR_DYNAMICS = False
_LEARN_GP_DYNAMICS = True

def train_and_save_svr_dynamics(train_file_base_name, svr_output_path,
                                delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                kernel_type='LINEAR',
                                kernel_params = {}):
    # Read data from disk
    print 'Reading training file:', train_file_base_name
    (train_X, train_Y) = dynamics_learning.read_dynamics_learning_example_files(train_file_base_name)
    # print 'train_X:\n',train_X
    # print 'train_Y:\n',train_Y
    print 'len(train_X)',len(train_X)
    print 'len(train_Y)',len(train_Y)
    # Build model class
    if _LEARN_SVR_ERR_DYNAMICS:
        svr_dynamics = SVRWithNaiveLinearPushDynamics(delta_t, n, m, epsilons=epsilons)
        svr_dynamics.learn_model(train_X, train_Y, kernel_params)
    elif _LEARN_GP_DYNAMICS:
        svr_dynamics = GPPushDynamics(delta_t, n, m,
                                      feature_names = feature_names,
                                      target_names = target_names,
                                      xtra_names = xtra_names)
        svr_dynamics.learn_model(train_X, train_Y)
    else:
        svr_dynamics = SVRPushDynamics(delta_t, n, m, epsilons=epsilons,
                                       feature_names = feature_names,
                                       target_names = target_names,
                                       xtra_names = xtra_names,
                                       kernel_type = kernel_type)
        # Train and save model
        svr_dynamics.learn_model(train_X, train_Y, kernel_params)

    print 'Saving model to:', svr_output_path
    svr_dynamics.save_models(svr_output_path)
    return svr_dynamics

def get_stats_for_diffs(error_i):
    error_stats = []
    # Mean and Standard deviation
    error_stats.append(np.mean(error_i))
    error_stats.append(np.sqrt(np.var(error_i)))

    # Ordinal stuff
    error_stats.append(np.min(error_i))
    [q1, median, q3] = np.percentile(error_i, [25, 50, 75])
    error_stats.append(q1)
    error_stats.append(median)
    error_stats.append(q3)
    error_stats.append(np.max(error_i))

    return error_stats

def analyze_pred_vs_gt(Y_hat, Y_gt, Angulars=[]):
    if len(Angulars) == 0:
        for i in xrange(len(Y_hat)):
            Angulars.append(False)

    Y_errors = []
    error_means = []
    error_sds = []
    error_mins = []
    error_q1s = []
    error_medians = []
    error_q3s = []
    error_maxes = []

    for Y_hat_i, Y_gt_i, is_angular in zip(Y_hat, Y_gt, Angulars):
        error_i = np.abs(np.array(Y_hat_i) - np.array(Y_gt_i))
        if is_angular:
            error_i = np.abs(util.subPIDiffNP(np.array(Y_hat_i), np.array(Y_gt_i)))
        Y_errors.append(error_i)
        # Get stats and add to specific lists
        error_i_stats = get_stats_for_diffs(error_i)
        error_means.append(error_i_stats.pop(0))
        error_sds.append(error_i_stats.pop(0))
        error_mins.append(error_i_stats.pop(0))
        error_q1s.append(error_i_stats.pop(0))
        error_medians.append(error_i_stats.pop(0))
        error_q3s.append(error_i_stats.pop(0))
        error_maxes.append(error_i_stats.pop(0))

    return (error_means, error_sds, Y_errors,
            error_mins, error_q1s, error_medians, error_q3s, error_maxes)

def test_svr_offline(model_param_file_name, test_data_example_base_name):
    delta_t = 1./9.
    n = 6
    m = 3
    if _LEARN_SVR_ERR_DYNAMICS:
        svr_dynamics = SVRWithNaiveLinearPushDynamics(delta_t, n, m, param_file_name = model_param_file_name)
    elif _LEARN_GP_DYNAMICS:
        svr_dynamics = GPPushDynamics(param_file_name = model_param_file_name)
    else:
        svr_dynamics = SVRPushDynamics(param_file_name = model_param_file_name)
    (test_X, test_Y) = dynamics_learning.read_dynamics_learning_example_files(test_data_example_base_name)
    # Run batch testing
    (Y_hat, Y_gt, _) = svr_dynamics.test_batch_data(test_X, test_Y)
    return (Y_hat, Y_gt)

def build_table_line(error_stats, test_obj_name):
    stat_str = ''
    for stat in error_stats:
        stat_str += str(stat) + '\t,'
    table_stat_line = test_obj_name + '\t,' + stat_str
    print table_stat_line
    table_stat_line += '\n'
    return table_stat_line

def write_stat_table(stat_name, table_lines, overall_line, title_line, header_line, table_out_path):
    table_lines.extend(overall_line)
    out_name = table_out_path + '-' + stat_name + '.csv'
    out_file = file(out_name, 'w')
    if len(title_line) > 0:
        out_file.write(title_line)
    if len(header_line) > 0:
        out_file.write(header_line)
    out_file.writelines(table_lines)
    out_file.close()

def build_results_table(error_means_all, error_sds_all, error_diffs_all,
                        error_mins_all, error_q1s_all, error_medians_all, error_q3s_all, error_maxes_all,
                        target_names, test_obj_names, hold_out_obj_name,
                        table_out_path=''):
    title_line = 'Hold out object: ' + hold_out_obj_name +'\t,'
    header_line = '\t,'
    for target_name in target_names:
        title_line += '\t,'
        header_line += target_name + '\t,'

    print title_line
    print header_line
    title_line += '\n'
    header_line += '\n'

    table_mean_lines = []
    table_sds_lines = []
    table_min_lines = []
    table_q1_lines = []
    table_median_lines = []
    table_q3_lines = []
    table_max_lines = []

    # Output to terminal
    for error_means, test_obj_name in zip(error_means_all, test_obj_names):
        table_mean_lines.append(build_table_line(error_means, test_obj_name))

    for error_sds, test_obj_name in zip(error_sds_all, test_obj_names):
        table_sds_lines.append(build_table_line(error_sds, test_obj_name))

    for error_mins, test_obj_name in zip(error_mins_all, test_obj_names):
        table_min_lines.append(build_table_line(error_mins, test_obj_name))

    for error_q1s, test_obj_name in zip(error_q1s_all, test_obj_names):
        table_q1_lines.append(build_table_line(error_q1s, test_obj_name))

    for error_medians, test_obj_name in zip(error_medians_all, test_obj_names):
        table_median_lines.append(build_table_line(error_medians, test_obj_name))

    for error_q3s, test_obj_name in zip(error_q3s_all, test_obj_names):
        table_q3_lines.append(build_table_line(error_q3s, test_obj_name))

    for error_maxes, test_obj_name in zip(error_maxes_all, test_obj_names):
        table_max_lines.append(build_table_line(error_maxes, test_obj_name))

    # Get mean and stand dev for each output dimension
    overall_means = []
    overall_sds = []
    overall_mins = []
    overall_q1s = []
    overall_medians = []
    overall_q3s = []
    overall_maxes = []

    # Group errors by output dimension
    overall_errors = []
    for i in xrange(len(target_names)):
        overall_errors.append(np.array([]))
        for error_diffs in error_diffs_all:
            overall_errors[i] = np.concatenate((overall_errors[i], error_diffs[i]))

    for overall_error in overall_errors:
        overall_stats = get_stats_for_diffs(overall_error)
        overall_means.append(overall_stats.pop(0))
        overall_sds.append(overall_stats.pop(0))
        overall_mins.append(overall_stats.pop(0))
        overall_q1s.append(overall_stats.pop(0))
        overall_medians.append(overall_stats.pop(0))
        overall_q3s.append(overall_stats.pop(0))
        overall_maxes.append(overall_stats.pop(0))

    overall_mean_line = build_table_line(overall_means, 'Overall')
    overall_sds_line = build_table_line(overall_sds, 'Overall')
    overall_min_line = build_table_line(overall_mins, 'Overall')
    overall_q1_line = build_table_line(overall_q1s, 'Overall')
    overall_median_line = build_table_line(overall_medians, 'Overall')
    overall_q3_line = build_table_line(overall_q3s, 'Overall')
    overall_max_line = build_table_line(overall_maxes, 'Overall')

    # Output to disk
    if len(table_out_path) > 0:
        if not os.path.exists(table_out_path):
            os.mkdir(table_out_path)
        table_out_path += hold_out_obj_name
        write_stat_table('means', table_mean_lines, overall_mean_line, title_line, header_line, table_out_path)
        write_stat_table('sds', table_sds_lines, overall_sds_line, title_line, header_line, table_out_path)

        write_stat_table('mins', table_min_lines, overall_min_line, title_line, header_line, table_out_path)
        write_stat_table('q1s', table_q1_lines, overall_q1_line, title_line, header_line, table_out_path)
        write_stat_table('medians', table_median_lines, overall_median_line, title_line, header_line, table_out_path)
        write_stat_table('q3s', table_q3_lines, overall_q3_line, title_line, header_line, table_out_path)
        write_stat_table('maxes', table_max_lines, overall_max_line, title_line, header_line, table_out_path)

def compare_obj_class_results(kernel_type = 'LINEAR', test_single_obj_model = True):
    target_names = [dynamics_learning._DELTA_OBJ_X_OBJ,
                    dynamics_learning._DELTA_OBJ_Y_OBJ,
                    dynamics_learning._DELTA_OBJ_THETA_WORLD,
                    dynamics_learning._DELTA_EE_X_OBJ,
                    dynamics_learning._DELTA_EE_Y_OBJ,
                    dynamics_learning._DELTA_EE_PHI_WORLD]

    hold_out_classes = _ALL_CLASSES[:]

    # Go through all objects
    if _LEARN_SVR_ERR_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/err_dyn/'
    elif _LEARN_GP_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/GP_DYN/'
    else:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/epsilon_e3/'
    base_example_dir_name = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/single_obj/'
    for hold_out_class in hold_out_classes:

        if test_single_obj_model:
            test_classes = _ALL_CLASSES[:]
            # test_classes.remove(hold_out_class)
        else:
            test_classes = [hold_out_class]
        if test_single_obj_model:
            model_param_file_name = base_svr_path + kernel_type + '_single_obj_' + hold_out_class + '_params.txt'
            if _LEARN_SVR_ERR_DYNAMICS:
                table_out_path = '/u/thermans/sandbox/single_objs/err_dyn/'
            elif _LEARN_GP_DYNAMICS:
                table_out_path = '/u/thermans/sandbox/single_objs/gp_dyn/'
            else:
                table_out_path = '/u/thermans/sandbox/single_objs/' + kernel_type +'/'
        else:
            model_param_file_name = base_svr_path + kernel_type + '_hold_out_' + hold_out_class + '_params.txt'
            if _LEARN_SVR_ERR_DYNAMICS:
                table_out_path = '/u/thermans/sandbox/hold_out_models/err_dyn/'
            elif _LEARN_GP_DYNAMICS:
                table_out_path = '/u/thermans/sandbox/hold_out_models/gp_dyn/'
            else:
                table_out_path = '/u/thermans/sandbox/hold_out_models/' + kernel_type +'/'
        if not os.path.exists(table_out_path):
            os.mkdir(table_out_path)

        # Test against object files for each of the held out objects independently
        Y_hat_all = []
        Y_gt_all = []
        error_means_all = []
        error_sds_all = []
        error_diffs_all = []
        error_mins_all = []
        error_q1s_all = []
        error_medians_all = []
        error_q3s_all = []
        error_maxes_all = []

        for test_obj in test_classes:
            if test_single_obj_model:
                print '\nTesting for object', test_obj, 'with model trained on', hold_out_class
            else:
                print '\nTesting for object', test_obj, 'with model trained on other classes'
            test_obj_example_base_name = base_example_dir_name + 'objs_' + test_obj
            (Y_hat, Y_gt) = test_svr_offline(model_param_file_name, test_obj_example_base_name)
            # Analyze output
            (error_means, error_sds, error_diffs,
             error_mins, error_q1s, error_medians, error_q3s, error_maxes) = analyze_pred_vs_gt(Y_hat, Y_gt)
            Y_hat_all.append(Y_hat)
            Y_gt_all.append(Y_gt)
            error_means_all.append(error_means)
            error_sds_all.append(error_sds)
            error_diffs_all.append(error_diffs)
            error_mins_all.append(error_mins)
            error_q1s_all.append(error_q1s)
            error_medians_all.append(error_medians)
            error_q3s_all.append(error_q3s)
            error_maxes_all.append(error_maxes)

        # Build table for data and save table to disk
        build_results_table(error_means_all, error_sds_all, error_diffs_all,
                            error_mins_all, error_q1s_all, error_medians_all, error_q3s_all, error_maxes_all,
                            target_names, test_classes, hold_out_class,
                            table_out_path)

def setup_leave_one_out_and_single_class_models(kernel_type = 'LINEAR', build_train_and_validate_data = False):
    train_classes = _ALL_CLASSES[:]

    # SVR options
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
        epsilons.append(1e-3)
    kernel_params = {}
    for i in xrange(len(target_names)):
        kernel_params[i] = '-g 0.05 -r 2'

    # Train and save models
    if _LEARN_SVR_ERR_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/err_dyn/'
    elif _LEARN_GP_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/GP_DYN/'
    else:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/epsilon_e3/'
    example_in_dir = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/'
    base_example_hold_out_dir_name = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/single_obj/'
    base_example_single_obj_dir_name = base_example_hold_out_dir_name

    for obj_class in train_classes:
        hold_out_classes = _ALL_CLASSES[:]
        hold_out_classes.remove(obj_class)

        if build_train_and_validate_data:
            print 'Creating holdout files for object:', obj_class
            dynamics_learning.create_train_and_validate_obj_class_splits(example_in_dir,
                                                                         base_example_hold_out_dir_name,
                                                                         hold_out_classes)
        # Train model with obj class as only left out class
        hold_out_str = ''
        for hold_out_class in hold_out_classes:
            hold_out_str += '_' + hold_out_class

        # hold_out_train_file_base_name = base_example_hold_out_dir_name + 'objs' + hold_out_str
        # svr_hold_out_output_path = base_svr_path + kernel_type + '_hold_out_' + obj_class

        # train_and_save_svr_dynamics(hold_out_train_file_base_name, svr_hold_out_output_path,
        #                             delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
        #                             kernel_type = kernel_type,
        #                             kernel_params = kernel_params)

        # Train model with obj class as only training class
        single_obj_train_file_base_name = base_example_single_obj_dir_name + 'objs_' + obj_class
        svr_single_obj_output_path = base_svr_path + kernel_type + '_single_obj_' + obj_class

        train_and_save_svr_dynamics(single_obj_train_file_base_name, svr_single_obj_output_path,
                                    delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                    kernel_type = kernel_type,
                                    kernel_params = kernel_params)

def build_shape_db(output_path, dynamics_model_name, shape_path, num_clusters):
    build_shape_db_exec = '../../bin/build_shape_db'
    run_switch = str(1)
    cmd = [build_shape_db_exec, run_switch, output_path, dynamics_model_name, shape_path, str(num_clusters)]
    cmd_str = ''
    for c in cmd:
        cmd_str += c + ' '
    print cmd_str
    p = subprocess.Popen(cmd)
    p.wait()

def build_object_class_shape_dbs(kernel_type='LINEAR'):
    example_in_dir = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/'
    base_shape_db_path = roslib.packages.get_pkg_dir('tabletop_pushing') + '/cfg/shape_dbs/'
    for obj_class in _ALL_CLASSES:
        hold_out_classes = _ALL_CLASSES[:]
        hold_out_classes.remove(obj_class)

        # Cluster local, global, and combined shape variables and write to each of the hold out object shape_db_files
        dynamics_model_name = kernel_type + '_single_obj_' + obj_class
        # local_shape_path = example_in_dir + obj_class + '_shape_local.txt'
        global_shape_path = example_in_dir + obj_class + '_shape_global.txt'
        # combined_shape_path = example_in_dir + obj_class + '_shape_combined.txt'
        num_clusters = 1

        for hold_out_class in hold_out_classes:
            # local_output_db_file = base_shape_db_path + 'hold_out_' + hold_out_class + '_local.txt'
            global_output_db_file = base_shape_db_path + 'hold_out_' + hold_out_class + '_global.txt'
            # combined_output_db_file = base_shape_db_path + 'hold_out_' + hold_out_class + '_combined.txt'

            # build_shape_db(local_output_db_file, dynamics_model_name, local_shape_path, num_clusters)
            build_shape_db(global_output_db_file, dynamics_model_name, global_shape_path, num_clusters)
            # build_shape_db(combined_output_db_file, dynamics_model_name, combined_shape_path, num_clusters)

def train_clustered_obj_model(obj_classes, kernel_type='LINEAR'):
    print 'obj_classes', obj_classes
    if _LEARN_SVR_ERR_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/err_dyn/'
    elif _LEARN_GP_DYNAMICS:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/GP_DYN/'
    else:
        base_svr_path = roslib.packages.get_pkg_dir('tabletop_pushing')+'/cfg/SVR_DYN/epsilon_e3/'

    example_in_dir = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/'
    out_dir_name = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/single_obj/'
    build_train_and_validate_data = True

    # SVR options
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
        epsilons.append(1e-3)
    kernel_params = {}
    for i in xrange(len(target_names)):
        kernel_params[i] = '-g 0.05 -r 2'

    print 'Building example file for objects:', obj_classes
    train_file_base_name = dynamics_learning.create_multi_obj_class_example_file(example_in_dir, out_dir_name,
                                                                                 obj_classes)

    # Train model with obj class as only left out class

    svr_output_path = base_svr_path + 'objs'
    for obj_class in obj_classes:
        svr_output_path += '_' + obj_class
    train_and_save_svr_dynamics(train_file_base_name, svr_output_path,
                                delta_t, n, m, epsilons, feature_names, target_names, xtra_names,
                                kernel_type = kernel_type,
                                kernel_params = kernel_params)


def parse_object_cluster_shape_db_file(file_path, obj_classes):
    cluster_file = file(file_path, 'r')
    cluster_names = [s.split(':')[0] for s in cluster_file.readlines()]
    cluster_file.close()
    class_labels = {}
    for i, name in enumerate(cluster_names):
        class_labels[i] = []
        # print name

    for obj_name in obj_classes:
        for i, cluster_name in enumerate(cluster_names):
            if obj_name in cluster_name:
                class_labels[i].append(obj_name)
    return class_labels

def cluster_shape_exemplars(obj_classes,
                            output_path, base_input_path, num_clusters, shape_suffix = '_shape_combined.txt'):
    build_shape_db_exec = '../../bin/build_shape_db'
    run_switch = str(0)
    cmd = [build_shape_db_exec, run_switch, output_path, base_input_path, str(num_clusters), shape_suffix]
    cmd.extend(obj_classes)
    cmd_str = ''
    for c in cmd:
        cmd_str += c + ' '
    print cmd_str
    p = subprocess.Popen(cmd)
    p.wait()

    class_labels = parse_object_cluster_shape_db_file(output_path, obj_classes)
    return class_labels

def train_shape_clusters(obj_classes, base_input_path,
                         output_path, num_clusters=5, shape_suffix = '_shape_global.txt'):
    # Make these parameters, so higher level functions can run this one
    class_labels = cluster_shape_exemplars(obj_classes, output_path, base_input_path, num_clusters,
                                           shape_suffix)
    # Build models based on these combined inputs
    for (key, value) in class_labels.items():
        print 'Cluster', key, 'has objects', value
        train_clustered_obj_model(value)

def write_rand_shape_db_file(class_labels, output_path):
    out_file = file(output_path, 'w')
    for (key, value) in class_labels.items():
        obj_str = 'objs'
        for obj_class in value:
            obj_str += '_' + obj_class
        if len(value) > 0:
            out_str = obj_str + ':' + str(key)
            print out_str
            out_file.write(out_str+'\n')
    out_file.close()

def train_random_object_class_clusters(obj_classes, shape_db_output_path, num_clusters = 5):
    class_labels = {}
    for i in xrange(num_clusters):
        class_labels[i] = []
    for obj_class in obj_classes:
        rand_idx = random.randint(0, num_clusters-1)
        class_labels[rand_idx].append(obj_class)

    # Build new shape db files form outputs
    write_rand_shape_db_file(class_labels, shape_db_output_path)

    for (key, value) in class_labels.items():
        print 'Cluster', key, 'has objects', value
        train_clustered_obj_model(value)

def train_hold_out_shape_clusters(num_clusters = 5, use_rand = False):
    obj_classes = _ALL_CLASSES[:]
    base_input_path = '/u/thermans/Dropbox/Data/rss2014/training/object_classes/'
    base_output_path = roslib.packages.get_pkg_dir('tabletop_pushing') + '/cfg/shape_dbs/'

    for obj_class in obj_classes:
        hold_out_classes = _ALL_CLASSES[:]
        hold_out_classes.remove(obj_class)

        if use_rand:
            output_path = base_output_path + 'rand2/hold_out_' + obj_class + '_global.txt'
            train_random_object_class_clusters(hold_out_classes, output_path, num_clusters)
        else:
            output_path = base_output_path + 'shape_cluster/hold_out_' + obj_class + '_global.txt'
            train_shape_clusters(hold_out_classes, base_input_path, output_path, num_clusters)

def learn_incremental_dynamics_models():
    obj_classes = _TEST_CLASSES
    base_output_path = roslib.packages.get_pkg_dir('tabletop_pushing') + '/cfg/SVR_DYN/incremental/'
    # TODO: Build incrememntal example files
    # TODO: Setup naming stuff
    num_trials = 0
    for obj_class in obj_classes:
        for i in num_trials:
            # TODO: train model with i trials
            output_path = base_output_path + 'single_obj_' + obj_class + '_' + i

def transform_target_into_obj_frame(x_1, x_0):
    # Get vector of x1 from x0 and roate the coordinates into the object frame
    x_1_demeaned = np.matrix([[ x_1[0] - x_0[0] ],
                              [ x_1[1] - x_0[1] ]])
    st = sin(x_0[2])
    ct = cos(x_0[2])
    R = np.matrix([[ ct, st],
                   [-st, ct]])
    x_1_obj = np.array(R*x_1_demeaned).T.ravel()
    delta_obj_x_obj = x_1_obj[0]
    delta_obj_y_obj = x_1_obj[1]
    delta_obj_theta_obj = util.subPIDiff(x_1[2], x_0[2])

    # Get ee vector and rotate into obj frame coordinates
    ee_0_demeaned = np.matrix([[x_0[3] - x_0[0]],
                               [x_0[4] - x_0[1]]])
    ee_1_demeaned = np.matrix([[x_1[3] - x_0[0]],
                               [x_1[4] - x_0[1]]])
    ee_0_obj = np.array(R*ee_0_demeaned).T.ravel()
    ee_1_obj = np.array(R*ee_1_demeaned).T.ravel()
    ee_delta_X = ee_1_obj - ee_0_obj
    delta_ee_x_obj = ee_delta_X[0]
    delta_ee_y_obj = ee_delta_X[1]

    delta_ee_phi_obj = util.subPIDiff(x_1[5], x_0[5])

    return np.array([delta_obj_x_obj,
                     delta_obj_y_obj,
                     delta_obj_theta_obj,
                     delta_ee_x_obj,
                     delta_ee_y_obj,
                     delta_ee_phi_obj])

def naive_model_trial_predictions(push_trials):
    n = 6
    m = 3
    dynamics = NaiveInputDynamics(1./9., n, m)
    Y_hat = []
    Y_gt = []
    for i in xrange(n):
        Y_hat.append([])
        Y_gt.append([])

    X_gt = []

    for i, trial in enumerate(push_trials):
        for j, step in enumerate(trial.trial_trajectory):
            [_, _, ee_phi] = tf.transformations.euler_from_quaternion(np.array([step.ee.orientation.x,
                                                                                step.ee.orientation.y,
                                                                                step.ee.orientation.z,
                                                                                step.ee.orientation.w]))
            x = np.array([step.x.x, step.x.y, step.x.theta,
                          step.ee.position.x, step.ee.position.y, ee_phi])
            u = np.array([step.u.linear.x, step.u.linear.y, step.u.angular.z])
            y_hat = dynamics.predict(x, u)

            if j < len(trial.trial_trajectory) - 1:
                delta_y_hat_obj = transform_target_into_obj_frame(y_hat, x)
                for k in xrange(n):
                    Y_hat[k].append(delta_y_hat_obj[k])
            if j > 0:
                delta_y_gt_obj = transform_target_into_obj_frame(x, x_prev)
                for k in xrange(n):
                    Y_gt[k].append(delta_y_gt_obj[k])
            # Update previous
            x_prev = x[:]
    return (Y_hat, Y_gt)

def get_aff_file_names(directory):
    all_files = os.listdir(directory)
    aff_files = []
    for f in all_files:
        if not (f.startswith('aff_learn_out') and not '-' in f):
            continue
        aff_files.append(directory + '/' + f)
    return aff_files

def analyze_naive_model_predicitons(base_input_path, table_out_path = '/u/thermans/sandbox/naive_linear/'):
    test_classes = _ALL_CLASSES[:]

    # TODO: Run through different test classes
    # TODO: Get aff_file_name from directories
    obj_dirs = os.listdir(base_input_path)
    aff_file_names = []
    for test_class in test_classes:
        for obj_dir in obj_dirs:
            if obj_dir.startswith(test_class):
                dir_aff_files = get_aff_file_names(base_input_path+obj_dir)
                aff_file_names.extend(dir_aff_files)
    # print 'Found aff files:'
    # for aff_file_name in aff_file_names:
    #     print aff_file_name

    target_names = ['Object X World',
                    'Object Y World',
                    'Object Theta World',
                    'EE X World',
                    'EE Y World',
                    'EE Phi World']

    Angular = [False, False, True, False, False, True]

    Y_hat_all = []
    Y_gt_all = []
    error_means_all = []
    error_sds_all = []
    error_diffs_all = []
    error_mins_all = []
    error_q1s_all = []
    error_medians_all = []
    error_q3s_all = []
    error_maxes_all = []

    all_errors = {}

    for aff_file_name in aff_file_names:
        plio = push_learning.CombinedPushLearnControlIO()
        plio.read_in_data_file(aff_file_name)

        (Y_gt, Y_hat) = naive_model_trial_predictions(plio.push_trials)

        # Analyze output
        (error_means, error_sds, error_diffs,
         error_mins, error_q1s, error_medians, error_q3s, error_maxes) = analyze_pred_vs_gt(Y_hat, Y_gt, Angular)

        Y_hat_all.append(Y_hat)
        Y_gt_all.append(Y_gt)

        error_means_all.append(error_means)
        error_sds_all.append(error_sds)
        error_diffs_all.append(error_diffs)
        error_mins_all.append(error_mins)
        error_q1s_all.append(error_q1s)
        error_medians_all.append(error_medians)
        error_q3s_all.append(error_q3s)
        error_maxes_all.append(error_maxes)

    # Build table for data and save table to disk
    build_results_table(error_means_all, error_sds_all, error_diffs_all,
                        error_mins_all, error_q1s_all, error_medians_all, error_q3s_all, error_maxes_all,
                        target_names, test_classes, 'Naive', table_out_path)
