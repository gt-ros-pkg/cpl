function symbols = calculate_symbol_distribution(m, symbols)
%CALCULATE_SYMBOL_DISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here


    for i=1:length(symbols)
        
        sd = [];
        ed = [];
        
        for g=m.g
            if g.id == i
                if isempty(sd)
                    sd = g.i_final.start_distribution * g.i_final.prob_notnull;
                    ed = g.i_final.end_distribution * g.i_final.prob_notnull;
                else
                    sd = sd + g.i_final.start_distribution * g.i_final.prob_notnull;
                    ed = ed + g.i_final.end_distribution * g.i_final.prob_notnull;
                end
            end
        end
        
        symbols(i).start_distribution = sd;
        symbols(i).end_distribution   = ed;
        
    end

end

