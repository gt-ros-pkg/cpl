function [slots] = gen_slots()

hand_off = 0.1;
df_reach_off = homo2d(-0.1, 0.0, 0.0);
df_table_width = 0.8;
df_bin_width = 0.1;
row1_off = 0.5;
row2_off = 0.9;
row3_offx = 0.9;
row3_offy = 1.1;
row3_offr = -pi/4.0;

slot_template = struct('row', -1, 'center', df_reach_off, 'reach_loc', df_reach_off, ...
                       'task_frame', df_reach_off);

slots(1:3+4*2) = slot_template;

row_nums_cell = num2cell(1*ones(1,3));
[slots(1:3).row] = row_nums_cell{:};
row1_center = homo2d(row1_off, 0.0, 0.0);
row_centers_cell = get_row_slots(3, df_table_width, df_bin_width, row1_center);
[slots(1:3).center] = row_centers_cell{:};

row_nums_cell = num2cell(2*ones(1,4));
[slots(4:7).row] = row_nums_cell{:};
row2_center = homo2d(row2_off, 0.0, 0.0);
row_centers_cell = get_row_slots(4, df_table_width, df_bin_width, row2_center);
[slots(4:7).center] = row_centers_cell{:};

row_nums_cell = num2cell(3*ones(1,4));
[slots(8:11).row] = row_nums_cell{:};
row3_center = homo2d(row3_offx, row3_offy, row3_offr);
row_centers_cell = get_row_slots(4, df_table_width, df_bin_width, row3_center);
[slots(8:11).center] = row_centers_cell{:};

for i = 1:numel(slots)
    reach_loc = slots(i).center * df_reach_off;
    slots(i).reach_loc = reach_loc;
    reach_hum_loc = homo2d(hand_off, 0.0, 0.0)^-1 * reach_loc;
    slots(i).task_frame = homo2d(hand_off, 0.0, atan2(reach_hum_loc(2,3),reach_hum_loc(1,3)));
    slots(i).hand_dist = norm([reach_hum_loc(1:2,3)]);
end

if 0
    all_centers = cell2mat({slots(:).center});
    all_reach_locs = cell2mat({slots(:).reach_loc});

    figure(2)
    clf
    hold on
    xlim([-1.5, 0.5])
    ylim([-0.5, 1.5])
    plot(-all_reach_locs(2,3:3:size(all_reach_locs,2))', all_reach_locs(1,3:3:size(all_reach_locs,2))', 'bx')
    plot(-all_centers(2,3:3:size(all_centers,2))', all_centers(1,3:3:size(all_centers,2))', 'mx')
    plot(0.0, hand_off, 'rx')

    p = [0.8*slots(3).hand_dist, 0.0, 1.0]';
    p2 = slots(3).task_frame * p;
    plot(-p2(2), p2(1), '+g')
end
