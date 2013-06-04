#!/usr/bin/env python

import os

TASK_PATH = 'tasks/'
BIN_PATH = 'tasks/bins/'
TASK_FILE_PRE = 'linear_chain_'
FILE_POST  = '.txt'

BIN_IDS_USED = []
POSSIBLE_BIN_IDS = [2,3,7,10,11,12,13,14,15,16,17,18,19,20]

def check_id(bin_to_check):
    global BIN_IDS_USED, POSSIBLE_BIN_IDS
    #check if bin id is valid
    if not bin_to_check in POSSIBLE_BIN_IDS:
        print 'Not a valid bin id.\nChoose one from 2,3,7 or 10-20'
        return False
    #check if bin id already in use
    if bin_to_check in BIN_IDS_USED:
        print 'Already used the given BIN ID.\nIDs already used= ' +str(BIN_IDS_USED)
        return False
    return True

if __name__ == '__main__':

    print 'CREATE NEW LINEAR PATH'
    
    file_exists = True
    while file_exists:
        lin_chain_num = int(raw_input('Linear Chain number: '))
        #check if file already exists
        file_exists = os.path.exists(TASK_PATH+ TASK_FILE_PRE+ str(lin_chain_num) + FILE_POST)
        if file_exists:
            print "The linear chain number you chose already exists."
            overwrite = raw_input('Overwrite (y/n) : ')
            if overwrite[0].lower() == 'y':
                file_exists = False

    no_of_bins = int(raw_input('Number of Bins to pick from: '))
    task_file = open(TASK_PATH+ TASK_FILE_PRE+ str(lin_chain_num) + FILE_POST, 'w')
    task_file.write('#sequence of bins to pick from for linear chain 1\n')

    pick_mean = float(raw_input('Mean time(in sec) to pick from a bin '))
    pick_var = float(raw_input('The variance = '))

    for i in range(no_of_bins):
        cur_bin_name = 'lc'+ str(lin_chain_num)+ '_b' + str(i+1)
        cur_bin_file = open(BIN_PATH + cur_bin_name + FILE_POST, 'w')

        task_file.write(cur_bin_name+'\n')
        cur_bin_file.write('#bin_id\n')
        
        can_use_id = False
        
        #check if chosen id is usable
        while not can_use_id:
            bin_id = int(raw_input('Bin ID for bin #'+str(i+1)+ ' : '))
            can_use_id = check_id(bin_id)
        
        BIN_IDS_USED.append(bin_id)

        cur_bin_file.write(str(bin_id) + '\n')
        cur_bin_file.write('#parts sequence: <part name> <mean> <std-dev>\n')
        
        num_parts = int(raw_input('No of parts in Bin#'+str(i+1)+' : '))
        
        for j in range(num_parts):
            cur_bin_file.write(cur_bin_name+ '_' + str(j+1) + ' ' + str(pick_mean) + ' ' + str(pick_var) + '\n')
        cur_bin_file.close()
    
    task_file.close()
    
    print "NEW LINEAR PATH TASK CREATED, CALLED- "  + TASK_FILE_PRE+ str(lin_chain_num)
    return
