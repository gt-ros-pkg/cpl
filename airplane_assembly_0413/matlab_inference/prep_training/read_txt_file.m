function frames = read_txt_file(file)


f = fopen(file, 'rt');

frames = struct([]);

while 1


    s = fgetl(f);
    %disp(s);

    if s(1) ~= 'F'
        break;
    end

    frames(end+1).s = s;
    s = textscan(s, '%s %d, %s %d, %s %d, %s %d, %s %d');
    
    frames(end).lefthand_msgnum = s{4};
    frames(end).righthand_msgnum = s{4};
    frames(end).bins_msgnum = s{4};
    frames(end).image_msgnum = s{4};
    
    % read hands
    frames(end).lefthand    = str2num(fgetl(f))';
    frames(end).righthand   = str2num(fgetl(f))';
    
    % read bins
    for i=1:20

        frames(end).bins(i).pq = [];
        frames(end).bins(i).H = [];
        
        bin = str2num(fgetl(f));

        if abs(1-norm(bin(4:7))) > 0.1
            continue;
        end;

        q = Quaternion(bin([7 4 5 6]));
        T = q.T;
        T(1:3,4) = bin(1:3)';
        
        frames(end).bins(i).pq = bin;
        frames(end).bins(i).H = T;
    end
end


fclose(f);



end