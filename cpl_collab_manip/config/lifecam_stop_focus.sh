#!/bin/bash
sudo uvcdynctrl -d video$1 --set="Focus, Auto" 0
sudo uvcdynctrl -d video$1 --set="Focus (absolute)" 0
