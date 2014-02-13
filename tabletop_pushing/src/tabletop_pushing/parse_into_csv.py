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

_STAT_NAMES = ['mean', 'std_dev', 'min', 'Q1', 'median', 'Q3', 'max',
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

def space_line_to_csv_line(line_in, add_leading_space = False):
    vals = line_in.split()
    line_out = ''
    for i, val in enumerate(vals):
        if (i == 0 and add_leading_space) or i > 0:
            line_out += ','
        line_out += val
    return line_out

def objectToObjectName(obj_name_raw):
    '''
    Replace underscores with spaces and captialize each word
    '''
    return ' '.join([obj_word.capitalize() for obj_word in obj_name_raw.split('_')])

def create_box_plot_csv(input_file_names, labels, metric, output_file_name):
    stats = {}
    for file_name, label in zip(input_file_names, labels):
        stats[label] = parse_stats_file(file_name)

    out_file = file(output_file_name, 'w')
    # TODO: Parse stats into output csv
    for label, data in stats.items():
        # print data.items()
        out_file.write( label + '\n')
        obj_line = 'stat'
        min_line = 'min'
        q1_line = 'Q1-min'
        med_line = 'median-Q1'
        q3_line = 'Q3-median'
        max_line = 'max-Q3'
        for obj in data.keys():
            obj_line +=  ', '+objectToObjectName(obj)
            min_line += ', '+str(data[obj][metric]['min'])
            q1_line += ', '+str(data[obj][metric]['Q1'] - data[obj][metric]['min'])
            med_line += ', '+str(data[obj][metric]['median'] - data[obj][metric]['Q1'])
            q3_line += ', '+str(data[obj][metric]['Q3'] - data[obj][metric]['median'])
            max_line += ', '+str(data[obj][metric]['max'] - data[obj][metric]['Q3'])
        out_file.write(obj_line+'\n')
        out_file.write(min_line+'\n')
        out_file.write(q1_line+'\n')
        out_file.write(med_line+'\n')
        out_file.write(q3_line+'\n')
        out_file.write(max_line+'\n')
    out_file.close()
