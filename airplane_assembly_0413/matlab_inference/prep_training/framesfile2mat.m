function [ output_args ] = framesfile2mat( framesfile, matfile )
%FRAMESFILE2MAT Summary of this function goes here
%   Detailed explanation goes here
    
    frames_info = read_txt_file(framesfile);
    save(matfile)

end

