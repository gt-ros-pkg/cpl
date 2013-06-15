import roslib
roslib.load_manifest("cpl_collab_manip")
import rospy
import yaml
from roslaunch.substitution_args import resolve_args

from project_simulation import msg.move_bin as move_bin_msg
from cpl_collab_manip.bin_manager import BinManager 

#if task is taking place currently or not
DOING_TASK = False

#callback for move-bin messages
def listen_task(task_msg):
    global bm, DOING_TASK
    DOING_TASK = False
    to_move_bin = task_msg.bin_id
    to_human_workspace = task_msg.move_near_human

    #locate the bin
    bin_found = False
    bin_list = bm.ar_man.get_available_bins()
    for bin_id in bin_list:
        if bin_id == to_move_bin:
            bin_found = True
            break
    if not bin_found():        
        print 'Could not complete task the task of moving bin-id: ', to_move_bin, ' because bin was not found.'
    else:
        if move_near_human:
            #check if bin already in workspace
            #check if there's an empty slot in the workspace
            #find the slot to move bin to
            bb()
        else:
            #check if bin already out-side the workspace
            #check if there's an empty-slot out of workspace
            #find which slot to move to
            bb()
        
        if move_now:
            #move to designated space
            if bm.move_bin():
                DOING_TASK = False
                return
            else:
                print 'Couldnot move bin: ', to_move_bin
                DOING_TASK = False
                return
        else:
            DOING_TASK = False
            return
    
#---------MAIN---------
def main():
    rospy.init_node("robot_mover")
    print 'Starting robot bin-mover...'
    arm_prefix = ""
    f_name = "$(find cpl_collab_manip)/config/bin_locs.yaml"
    f = file(resolve_args(f_name))
    bin_slots = yaml.load(f)['data']
    bm = BinManager(arm_prefix, bin_slots)
    
    #reset bin manager system first
    print 'Commence system reset'
    bm.system_reset()
    
    #setup listening
    task_listen_sub = rospy.Subscriber('move_bin', project_simulation.msg.move_bin, listen_task)

    

if __name__=='__main__':
    main()
