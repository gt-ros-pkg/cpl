#! /usr/bin/python
import rospy
import roslib
roslib.load_manifest('project_simulation')

import tf

import numpy
import random
import task_description
import math

if __name__ == '__main__':

    t       = task_description.taskfile2dict('model.xml')
    
    for a_symbol in t['grammar']['symbols']:
        if a_symbol['is_terminal'] == 0 and not a_symbol['name']=='S' :
            #print a_symbol['name'] + ' ' + str(a_symbol['detector_id'])
            bin_file = open("bins/"+a_symbol['name']+".txt", 'w')
            bin_file.write('#bin_id\n')
            first = True
            for term_symbol in t['grammar']['symbols']:
                if term_symbol['is_terminal']==1 and term_symbol['name'][:-1] == a_symbol['name'].lower():
                    print 'reached'
                    if first:
                        bin_file.write(str(int(term_symbol['detector_id']))+'\n')
                        bin_file.write("#parts sequence: <part name> <mean> <std-dev>\n")
                        first = False
                    
                    bin_file.write(term_symbol['name']+' '+str(term_symbol['learntparams']['duration_mean']/30)+' '+ str(math.sqrt(term_symbol['learntparams']['duration_var']/(30.0*30.0)))+'\n')
    

