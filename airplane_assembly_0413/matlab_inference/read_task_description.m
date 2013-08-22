function [task] = read_task_description(taskfilename)
dir = '~/dev/gt_ros_pkg_cpl/project_simulation/src/tasks/';
taskfile = strsplit(fileread(strcat(dir,taskfilename,'.txt')),'\n');
task = {};
bins = taskfile(2:end-1)
for i = 1:numel(bins)
    binfilename = strcat(dir,'bins/',bins(i),'.txt');
    binfile = strsplit(fileread(binfilename{1}),'\n');
    task{end+1} = struct;
    task{end}.draw_ids = {};
    task{end}.draw_means = [];
    task{end}.draw_stds = [];
    task{end}.bin_id = str2num(binfile{2});
    draws = binfile(4:end-1);
    for j = 1:numel(draws)
        draw = strsplit(draws{j},' ');
        task{end}.draw_ids{end+1} = draw{1};
        task{end}.draw_means(end+1) = str2num(draw{2});
        task{end}.draw_stds(end+1) = str2num(draw{3});
    end
    task{end}
end

