#! /usr/bin/python

import rospy
import roslib
roslib.load_manifest('utilities_aa0413')
import tf
import numpy
from lxml import etree
import random
import math
####################################################
# var
####################################################

####################################################
# functions
####################################################

def taskfile2dict(xmlfile):
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    return element2dict(root)

def element2dict(e):

    if e.text is None:
        r = {}

        for c in e:
            if not r.has_key(c.tag):
                 r[c.tag] = []

            r[c.tag].append(  element2dict(c)  )

        for k in r.keys():
            if len(r[k]) == 1:
                 r[k] = r[k][0]

        if len(r) == 0:
            r = None

    else:
        r = e.text

        if e.attrib.has_key('rows') and e.attrib.has_key('cols'):
             rows = float(e.attrib['rows'])
             cols = float(e.attrib['cols'])
             if rows == 1 and cols == 1:
                 r = float(r)
             else:
                 r = numpy.fromstring(r, dtype=float, sep=' ')
                 r = numpy.reshape(r, (rows,cols)) 
    return r
       


def gen_random_task_by_symbolid(sid, task):

    actions = []

    s = task['grammar']['symbols'][sid-1]

    r = None
    if not s['rule_id'] is None:
        r = task['grammar']['rules'][int(s['rule_id']-1)]

    if s['is_terminal']:
        
        dur = random.gauss(s['learntparams']['duration_mean'], math.sqrt(s['learntparams']['duration_var']))
        return [{'name': s['name'], 'bin_id': int(s['detector_id']), 'dur': dur}]

    elif r['or_rule'] == 1 :
        i = random.randint(0, r['right'].shape[0] * r['right'].shape[1] - 1);
        return gen_random_task_by_symbolid(int(r['right'].item(i)), task)

    else:

        for i in numpy.nditer(r['right']):
            actions = actions + gen_random_task_by_symbolid(int(i), task)

    return actions

def gen_random_task(task):
    random.seed()
    return gen_random_task_by_symbolid(int(task['grammar']['starting']), task)


####################################################
# test
####################################################

def main():
    print 'nothing'

if __name__ == '__main__' :
    main()























































































