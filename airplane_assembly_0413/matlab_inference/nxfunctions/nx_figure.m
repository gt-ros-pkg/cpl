function h = nx_figure( i )
%NX_FIGURE Summary of this function goes here
%   Detailed explanation goes here

    h = -1;
    
    try
        set(0,'CurrentFigure', i)
    catch
        h = figure(i)
    end

end

