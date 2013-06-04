function [] = visualize_bin_probs(t, numbins, probs, tnow, maxtime)

hold on
axis([0, maxtime, 0, numbins])
for i = 1:numbins
    maxprob = 1.1*max([probs{i,1}, probs{i,2}]);
    plot(t,probs{i,1}/maxprob-i+numbins,'r')
    plot(t,probs{i,2}/maxprob-i+numbins,'b')
    plot(t,zeros(1,numel(t))-i+numbins,'k')
end

ylabels = {};
for i = 1:numbins
    ylabels{i} = sprintf('Bin %d', numbins-i+1);
end
plot([tnow, tnow], [0, numbins], 'g');

AX = gca;
set(AX,'YTick',(1:numbins)-0.5);
set(AX,'YTickLabel',ylabels);
set(AX,'YGrid','off');
set(AX,'XGrid','on');
