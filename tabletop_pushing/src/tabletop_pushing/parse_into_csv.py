#!/usr/bin/env python

_POS_ERROR_HEADER = '# Position error'
_PUSH_TIME_HEADER = '# Push time'
_PUSH_DIST_HEADER = '# Push dist'
_DELTA_THETA_HEADER = '# delta theta'
_AVG_VEL_HEADER = '# avg velocities'

_BREAKDOWN_HEADER = '# mean std_dev min Q1 median q3 max [sub2 sub5 total]'

_POS_ERROR_METRIC = 'pos_error'
_PUSH_TIME_METRIC = 'push_time'
_PUSH_DIST_METRIC = 'push_dist'
_DELTA_THETA_METRIC = 'delta_theta'
_AVG_VEL_METRIC = 'avg_vel'

_METRIC_HEADERS = {_POS_ERROR_HEADER : _POS_ERROR_METRIC,
                   _PUSH_TIME_HEADER : _PUSH_TIME_METRIC,
                   _PUSH_DIST_HEADER : _PUSH_DIST_METRIC,
                   _DELTA_THETA_HEADER : _DELTA_THETA_METRIC,
                   _AVG_VEL_HEADER : _AVG_VEL_METRIC}

_STAT_NAMES = ['mean', 'std_dev', 'min', 'Q1', 'median', 'q3', 'max',
               'sub2', 'sub5', 'total']

def get_stats_dict(stat_line):
    stat_dict = {}
    for i, d in enumerate(stat_line.split()):
        d_flt = float(d)
        if d_flt - round(d_flt) == 0.0:
            d_flt = int(d_flt)
        stat_dict[ _STAT_NAMES[i] ] = d_flt
    return stat_dict

def parse_stats_file(file_name):
    file_in = file(file_name, 'r')
    data_in = file_in.readlines()
    file_in.close()
    obj_results = {}
    cur_obj_name = None
    next_line_data = False
    for line in data_in:
        line = line.rstrip()
        if line.startswith('#'):
            if line in _METRIC_HEADERS.keys():
                next_line_data = False
                cur_data_key = _METRIC_HEADERS[line]
            elif line.startswith(_BREAKDOWN_HEADER):
                next_line_data = True
            else:
                cur_obj_name = line[1:].strip()
                next_line_data = False
                obj_results[cur_obj_name] = {}
        elif next_line_data:
            next_line_data = False
            data = get_stats_dict(line)
            obj_results[cur_obj_name][cur_data_key] = data
        else:
            print 'Nothing to do with line', line
            print 'Previous line was', line
        prev_line = line
    return obj_results

def group_stats_from_files(file_names, labels, metric):
    # TODO: Read in files
    # TODO: Parse stats
    # TODO: Select the desired stats
    pass

def create_box_plot_csv(file_names, labels, metric):
    pass
