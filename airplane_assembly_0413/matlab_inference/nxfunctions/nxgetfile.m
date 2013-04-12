function file = nxgetfile()
%NXGETFILE Summary of this function goes here
%   Detailed explanation goes here

    [file path] = uigetfile('*'); file = [path file];

end

