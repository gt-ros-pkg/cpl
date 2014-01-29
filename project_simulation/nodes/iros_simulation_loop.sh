#!/bin/bash

for i in {1..25}
do
   echo "(((((((((((((((((((Welcome $i times))))))))))))))))"

        sleep 5

	rosrun project_simulation fix_tf.py &
	PID1=$!

	rosrun project_simulation ar_markers_publish.py &
	PID2=$!

	rosrun project_simulation robo_sim_listen.py &
	PID3=$!

    echo y | rosrun project_simulation hands_sim iros_a2 n $i 0.01 &
	PID4=$!

	rosrun airplane_assembly_inference_0313 planning_from_matlab.py &
	PID5=$!

	rosrun airplane_assembly_inference_0313 inference_from_matlab.py

	kill $PID1
	kill $PID2
	kill $PID3
	kill $PID4
	kill $PID5


	echo END

        sleep 5

done

for i in {1..25}
do
   echo "(((((((((((((((((((Welcome $i times))))))))))))))))"

        sleep 5

	rosrun project_simulation fix_tf.py &
	PID1=$!

	rosrun project_simulation ar_markers_publish.py &
	PID2=$!

	rosrun project_simulation robo_sim_listen.py &
	PID3=$!

    echo y | rosrun project_simulation hands_sim iros_a2 n $i 0.2 &
	PID4=$!

	rosrun airplane_assembly_inference_0313 planning_from_matlab.py &
	PID5=$!

	rosrun airplane_assembly_inference_0313 inference_from_matlab.py

	kill $PID1
	kill $PID2
	kill $PID3
	kill $PID4
	kill $PID5


	echo END

        sleep 5

done

for i in {1..25}
do
   echo "(((((((((((((((((((Welcome $i times))))))))))))))))"

        sleep 5

	rosrun project_simulation fix_tf.py &
	PID1=$!

	rosrun project_simulation ar_markers_publish.py &
	PID2=$!

	rosrun project_simulation robo_sim_listen.py &
	PID3=$!

    echo y | rosrun project_simulation hands_sim iros_h n $i 0.01 &
	PID4=$!

	rosrun airplane_assembly_inference_0313 planning_from_matlab.py &
	PID5=$!

	rosrun airplane_assembly_inference_0313 inference_from_matlab.py

	kill $PID1
	kill $PID2
	kill $PID3
	kill $PID4
	kill $PID5


	echo END

        sleep 5

done

for i in {1..25}
do
   echo "(((((((((((((((((((Welcome $i times))))))))))))))))"

        sleep 5

	rosrun project_simulation fix_tf.py &
	PID1=$!

	rosrun project_simulation ar_markers_publish.py &
	PID2=$!

	rosrun project_simulation robo_sim_listen.py &
	PID3=$!

    echo y | rosrun project_simulation hands_sim iros_h n $i 0.2 &
	PID4=$!

	rosrun airplane_assembly_inference_0313 planning_from_matlab.py &
	PID5=$!

	rosrun airplane_assembly_inference_0313 inference_from_matlab.py

	kill $PID1
	kill $PID2
	kill $PID3
	kill $PID4
	kill $PID5


	echo END

        sleep 5

done