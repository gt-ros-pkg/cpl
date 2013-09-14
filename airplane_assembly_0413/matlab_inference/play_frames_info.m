nx_figure(1);
axis equal;
xlim([-1.5 0.5])
ylim([-1.5 0.5])

for frame_info = frames_info
    cla;
    hold on;
    plot(frame_info.lefthand(1), frame_info.lefthand(2), '*r');
    plot(frame_info.righthand(1), frame_info.righthand(2), '*r');

    for b=0:length(m.detection.detectors)-1
      if ~isempty(frame_info.bins(b+1).H)
        plot(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), '.b');
        text(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), num2str(b+1)); 

        d = max(norm([frame_info.righthand(1), frame_info.righthand(2)] - [frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2)]), norm([frame_info.lefthand(1), frame_info.lefthand(2)] - [frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2)]));
        if d < 0.7
            plot(frame_info.bins(b+1).pq(1), frame_info.bins(b+1).pq(2), '+g');
        end
      end
    end

    hold off
    
    pause(0.2);
end