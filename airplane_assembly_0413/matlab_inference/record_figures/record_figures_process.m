function data = record_figures_process( data )
%RECORD_FIGURES_PROCESS Summary of this function goes here
%   Detailed explanation goes here

    figHandles = findobj('Type', 'figure');
    
    for f=figHandles'
        
        try
            
            %nx_figure(f);
            %currFrame = getframe(gcf);
            %currFrame = getframe(get(0,'CurrentFigure'));
            %currFrame = getframe(f);
            
            currFrame = getframe_nosteal_focus(f);
            
            %writeVideo(data.vidObj{f},currFrame);
            
            writeVideo(data.vidObj{f},currFrame.cdata);
            
            
            
        catch
            
            disp([data.path data.name '_' num2str(f) '.avi']);
            
            data.vidObj{f} = VideoWriter([data.path data.name '_' num2str(f) '.avi']);
            data.vidObj{f}.FrameRate = data.framerate;
            open(data.vidObj{f});
            
            %nx_figure(f);
            %currFrame = getframe(gcf);
            %currFrame = getframe(get(0,'CurrentFigure'));
            %currFrame = getframe(f);
            
            currFrame = getframe_nosteal_focus(f);
            
            %writeVideo(data.vidObj{f},currFrame);
            
            writeVideo(data.vidObj{f},currFrame.cdata);
            
        end
        
    end

end

