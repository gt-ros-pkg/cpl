
function [] = m_plot_distributions(m, start_symbols, end_symbols, scale_max)

    if ~exist('scale_max')
        scale_max = 1;
    end

    cla
    hold on;
    for action = start_symbols
        d = m.grammar.symbols(actionname2symbolid(action{1}, m.grammar)).start_distribution;
        if scale_max
            d = d / max(d) * sum(d);
        end
        plot(d, 'color', nxtocolor(sum(action{1})));
    end
    for action = end_symbols
        d = m.grammar.symbols(actionname2symbolid(action{1}, m.grammar)).end_distribution;
        if scale_max
            d = d / max(d) * sum(d);
        end
        plot(d, '--', 'color', nxtocolor(sum(action{1})));
    end
    legend([start_symbols, end_symbols]);
    %plot(nt, 0, '*black');
    hold off;

end