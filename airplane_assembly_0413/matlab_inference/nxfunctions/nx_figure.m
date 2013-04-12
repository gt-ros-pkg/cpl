function [ output_args ] = nx_figure( i )
%NX_FIGURE Summary of this function goes here
%   Detailed explanation goes here


    try
        set(0,'CurrentFigure', i)
    catch
        figure(i)
    end

end

