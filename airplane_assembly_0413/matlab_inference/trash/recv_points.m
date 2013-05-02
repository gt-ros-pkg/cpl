

t = tcpip('localhost', 12345);
t.OutputBufferSize = 8 * 1000;

fopen(t);

while 1
    
    while t.BytesAvailable < 10 + 8 * 3
        pause(0.01);
    end
    
    name = fread(t, 10, 'char');
    name = char(name');
    data = fread(t, 3, 'double');
    
    disp(name);
    disp(data');
    
end


















