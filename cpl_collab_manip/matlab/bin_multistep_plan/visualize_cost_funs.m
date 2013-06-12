function [] = visualize_cost_funs(t, probs, undo_dur, nowtimesec, nowtimeind, maxtime)
numbins = size(probs,1);

%axis([0, maxtime, -10, 100])
max_costs = 100;
hold on
axis([0, maxtime, 0, numbins])
for i = 1:numbins
    binprob = sum(probs{i,1});
    for tx = 1:numel(t)
        rm_costs(tx) = remove_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, 1, undo_dur);
        lt_costs(tx) = late_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, nowtimeind);
    end
    rm_costs(rm_costs > max_costs) = nan;
    lt_costs(lt_costs > max_costs) = nan;
    %yval = numbins-(i-1);
    plot(t, rm_costs/max_costs/1.1-i+numbins, 'r')
    plot(t, lt_costs/max_costs/1.1-i+numbins, 'b')
    plot(t,zeros(1,numel(t))-i+numbins,'k')
end
plot([nowtimesec, nowtimesec], [0, numbins], 'g');
