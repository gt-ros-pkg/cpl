function [plan] = optimize_time(slot_states, deliv_seq)

cur_deliv = 1;
plan = [];

while 1
    empty_slot = 0;
    for i = 1:numel(slot_states)
        if slot_states(i) < 0
            empty_slot = i;
            break
        end
    end
    if empty_slot > 0
        % have empty slot, fill it
        fill_bin = deliv_seq(cur_deliv);
        plan(end+1) = fill_bin;
        slot_states(empty_slot) = fill_bin;
        cur_deliv = cur_deliv + 1;
        if cur_deliv > numel(deliv_seq)
            % filled all bins, we're done!
            break
        end
    else
        % no empty slot, must remove one
        plan(end+1) = -1;
        slot_states(1) = -1;
    end
end
