

PORT_NUMBER = 12345;
BIN_NUM     = 20;



t = tcpip('localhost', PORT_NUMBER);
t.OutputBufferSize = 8 * 1000;
t.InputBufferSize  = 8 * 1000;
fopen(t)


%%
while 1
    
    newdata = 0
    
    % get data
    while t.BytesAvailable >= 4 * (2 * 3 + BIN_NUM * 7)
        
        disp received
        
        newdata = 1
        
        v = fread(t, 2 * 3 + BIN_NUM * 7, 'float');
        
    end
    
    % draw
    if newdata
        
        lefthand = v(1:3);
        righthand = v(4:6);

        vv = v(7:end);

        cla;
        hold on;
        plot(-2, -2, '*');
        plot(2, 2, '*');
        plot(lefthand(1), lefthand(2), '*r');
        plot(righthand(1), righthand(2), '*r');

        for b=0:BIN_NUM-1
          if norm(b) > 0.001
            bin = vv(b*7 + 1: b*7 + 7);
            plot(bin(1), bin(2), '.g');
          end
        end

        hold off
        
    end
    pause(0.01)
end











