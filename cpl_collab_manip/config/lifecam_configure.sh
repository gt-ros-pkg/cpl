#!/bin/bash
sudo uvcdynctrl -d video$1 --set="Focus, Auto" 0
sudo uvcdynctrl -d video$1 --set="Focus (absolute)" 0
sudo uvcdynctrl -d video$1 -s "White Balance Temperature" 2800
sudo uvcdynctrl -d video$1 -s "White Balance Temperature, Auto" 0
sudo uvcdynctrl -d video$1 -s "Brightness" 70
