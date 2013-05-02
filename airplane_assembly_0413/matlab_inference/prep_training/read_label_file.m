
function labels = read_label_file(file)

disp(['Read label file ' file]);

labels = struct('name', {}, 'start', {}, 'end', {});

labelfile = fopen(file, 'rt');

while 1
    
    s = fgetl(labelfile);
    
    if s == -1, break; end;
    if length(s) < 3 || s(1) == ' ', continue; end
    
    disp(s);
    
    s = textscan(s, '%s %f %f');
    labels(end+1) = struct('name', s{1}, 'start', s{2}, 'end', s{3});
    
end

for i=1:length(labels)
    
    if labels(i).end == 0,
        labels(i).end = labels(i+1).start - 1;
    end
    
    assert(labels(i).start <= labels(i).end);
    
    if i > 1,
        assert(labels(i).start > labels(i-1).end);
    end
end

fclose(labelfile);


end







































