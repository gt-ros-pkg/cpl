function [] = visualize_bin_probs(t, numbins, probs, bin_names, bin_relevs, ...
                                  nowtimesec, nowtimeind, maxtime)

hold on
axis([0, maxtime, 0, numbins])
[~, bin_ranks] = sort(bin_relevs,2,'descend');
ylabels = {};
for i = 1:numbins
    maxprob = max([probs{i,1}, probs{i,2}]);
    binprob = sum(probs{i,1})
    plot(t,probs{i,1}/(1.1*maxprob)-i+numbins,'b')
    plot(t,probs{i,2}/(1.1*maxprob)-i+numbins,'r')
    plot(t,zeros(1,numel(t))-i+numbins,'k')
    plot([maxtime-10, maxtime],(binprob/1.1-i+numbins)*ones(1,2),'m')
    text(nowtimesec+3, 0.5-i+numbins, sprintf('%d (%.1f)', bin_ranks(i), bin_relevs(i)));
    bin_prob = sum(probs{i,1});
    start_probs = probs{i,1}/bin_prob;
    end_probs = probs{i,2}/bin_prob;
    prob_now_before_bin = sum(start_probs(nowtimeind+1:end));
    prob_now_after_bin = sum(end_probs(1:nowtimeind));
    prob_now_during_bin = (1-prob_now_before_bin) * (1-prob_now_after_bin);
    text(maxtime-10, 0.5-i+numbins, sprintf('B:%1.2f, D:%1.2f, A:%1.2f', ...
                                prob_now_before_bin, prob_now_during_bin, prob_now_after_bin));
    ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
end
plot([nowtimesec, nowtimesec], [0, numbins], 'g');

AX = gca;
set(AX,'YTick',(1:numbins)-0.5);
set(AX,'YTickLabel',ylabels);
set(AX,'YGrid','off');
set(AX,'XGrid','on');
