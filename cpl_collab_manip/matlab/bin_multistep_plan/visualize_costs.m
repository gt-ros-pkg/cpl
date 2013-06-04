planning_params

figure(2)
clf
subplot(3,1,1)
hold on
maxprob = 0;
axis([0, t(100*rate), 0, 0.08])
for i = 1:numbins
    plot(t,probs{i,1},'r')
    plot(t,probs{i,2},'b')
    maxprob = max([probs{i,1}, probs{i,2}, maxprob]);
end
hold on
plot([nowtimesec, nowtimesec], [0, maxprob], 'g');

rm_costs = [];
lt_costs = [];
for i = 1:numbins
    for tx = 1:numel(t)
        binprob = sum(probs{i,1});
        rm_costs(i,tx) = remove_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, undodur);
        lt_costs(i,tx) = late_cost(t, tx, probs{i,1}/binprob, probs{i,2}/binprob, binprob, nowtimeind);
    end
end

subplot(3,1,2)
hold on
ylabel('Remove Costs')
%axis([0, t(end), 0, 100])
axis([0, t(100*rate), 0, 100])
for i = 1:numbins
    plot(t, rm_costs(i,:), 'r')
end

subplot(3,1,3)
hold on
ylabel('Late Costs')
axis([0, t(100*rate), 0, 100])
for i = 1:numbins
    plot(t, lt_costs(i,:), 'b')
end
