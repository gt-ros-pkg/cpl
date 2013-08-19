function [] = visualize_cost_funs(t, probs, is_delivered, undo_dur, undo_dur_ind, nowtimesec, nowtimeind, maxtime)
numbins = size(probs,1);

%axis([0, maxtime, -10, 100])
max_costs = 100;
hold on
axis([0, maxtime, 0, numbins])
for i = 1:numbins
    binprob = sum(probs{i,1});
    % for tx = 1:numel(t)
    %     rm_costs(tx) = remove_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, 1, undo_dur);
    %     lt_costs(tx) = late_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, nowtimeind);
    % end
    rm_costs_new = remove_cost_precomp(t, probs{i,1}/binprob, probs{i,2}/binprob, binprob, undo_dur, is_delivered(i));
    redeliv_costs_new = redeliv_cost_precomp(t, probs{i,1}/binprob, probs{i,2}/binprob, binprob, undo_dur, is_delivered(i), undo_dur_ind);
    lt_costs_new = late_cost_precomp(t, probs{i,1}/binprob, probs{i,2}/binprob, binprob, nowtimeind, nowtimeind, undo_dur_ind);
    % mean((lt_costs(nowtimeind:end)-lt_costs_new(nowtimeind:end)).^2)
    % rm_costs(rm_costs > max_costs) = nan;
    % lt_costs(lt_costs > max_costs) = nan;
    rm_costs_new(rm_costs_new > max_costs) = nan;
    redeliv_costs_new(rm_costs_new > max_costs) = nan;
    lt_costs_new(lt_costs_new > max_costs) = nan;
    %yval = numbins-(i-1);
    % plot(t, rm_costs/max_costs/1.1-i+numbins, 'r')
    % plot(t, lt_costs/max_costs/1.1-i+numbins, 'b')
    plot(t, rm_costs_new/max_costs/1.1-i+numbins, 'r')
    plot(t, redeliv_costs_new/max_costs/1.1-i+numbins, 'g')
    plot(t, lt_costs_new/max_costs/1.1-i+numbins, 'b')
    plot(t,zeros(1,numel(t))-i+numbins,'k')
end
plot([nowtimesec, nowtimesec], [0, numbins], 'g');
