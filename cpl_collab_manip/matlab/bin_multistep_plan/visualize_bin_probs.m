function [] = visualize_bin_probs(t, numbins, probs, bin_names, bin_relevs, tnow, maxtime)

hold on
axis([0, maxtime, 0, numbins])
[~, bin_ranks] = sort(bin_relevs,2,'descend');
for i = 1:numbins
    maxprob = max([probs{i,1}, probs{i,2}]);
    binprob = sum(probs{i,1})
    plot(t,probs{i,1}/(1.1*maxprob)-i+numbins,'r')
    plot(t,probs{i,2}/(1.1*maxprob)-i+numbins,'b')
    plot(t,zeros(1,numel(t))-i+numbins,'k')
    plot([maxtime-10, maxtime],(binprob/1.1-i+numbins)*ones(1,2),'m')
    text(tnow+3, 0.5-i+numbins, sprintf('%d (%.1f)', bin_ranks(i), bin_relevs(i)));
end

ylabels = {};
for i = 1:numbins
    ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
end
plot([tnow, tnow], [0, numbins], 'g');

AX = gca;
set(AX,'YTick',(1:numbins)-0.5);
set(AX,'YTickLabel',ylabels);
set(AX,'YGrid','off');
set(AX,'XGrid','on');
