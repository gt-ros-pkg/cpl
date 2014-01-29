function [] = visualize_bin_probs(t, numbins, probs, bin_names, bin_relevs, ...
                                  nowtimesec, nowtimeind, maxtime, for_humanoids)

hold on
axis([0, maxtime, 0, numbins])
[~, bin_ranks] = sort(bin_relevs,2,'descend');
ylabels = {};
for i = 1:numbins
    maxprob = max([probs{i,1}, probs{i,2}]);
    binprob = sum(probs{i,1});
    plot(t,probs{i,1}/(1.1*maxprob)-i+numbins,'b')
    plot(t,probs{i,2}/(1.1*maxprob)-i+numbins,'r')
    plot(t,zeros(1,numel(t))-i+numbins,'k')

    if ~for_humanoids
        plot([maxtime-10, maxtime],(binprob/1.1-i+numbins)*ones(1,2),'m')
        text(nowtimesec+3, 0.5-i+numbins, sprintf('%d (%.1f)', find(bin_ranks==i), bin_relevs(i)));
    end
    
    bin_prob = sum(probs{i,1});
    start_probs = probs{i,1}/bin_prob;
    end_probs = probs{i,2}/bin_prob;
    prob_now_before_bin = sum(start_probs(nowtimeind+1:end));
    prob_now_after_bin = sum(end_probs(1:nowtimeind));
    prob_now_during_bin = (1-prob_now_before_bin) * (1-prob_now_after_bin);
    
    if ~for_humanoids
        text(maxtime-60, 0.8-i+numbins, sprintf('P: %1.2f, B:%1.2f, D:%1.2f, A:%1.2f', ...
                    binprob, prob_now_before_bin, prob_now_during_bin, prob_now_after_bin));
    end
    if ~for_humanoids
        ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
    else
        ylabels{i} = sprintf('Bin %s', bin_names{numbins-i+1});
        % ylabels{i} = sprintf('Bin %d', numbins-i+1);
    end
end

if for_humanoids
    text(nowtimesec+0.5, numbins-0.2, sprintf('Now'));
    %set(gca, 'FontWeight', 'demi');
end
plot([nowtimesec, nowtimesec], [0, numbins], 'g');

if 0
    hleg = legend('Start','End');
    set(hleg, 'FontName', 'Times', 'FontSize', 14)
    set(hleg,'Location','NorthEast')
    set(hleg,'Interpreter','none')
end
AX = gca;
set(AX,'YTick',(1:numbins)-0.5);
set(AX,'YTickLabel',ylabels);
set(AX,'XTickLabel','');
set(get(AX, 'YLabel'), 'String', 'Bin Demand Distributions');
set(AX,'YGrid','off');
set(AX,'XGrid','on');
