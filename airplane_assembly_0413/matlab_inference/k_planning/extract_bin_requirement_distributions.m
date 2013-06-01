function bin_distributions = extract_bin_requirement_distributions( m )
%EXTRACT_BIN_REQUIREMENT_DISTRIBUTIONS Summary of this function goes here
%   Detailed explanation goes here

bin_id        = {3, 11, 10, 12, 2, 7, 14, 15, 13};
symbol_names1 = {'body1', 'nose_a1', 'nose_h1', 'wing_at1', 'wing_ad1', 'wing_h1', 'tail_at1', 'tail_ad1', 'tail_h1'};
symbol_types1 = {'start', 'start', 'start', 'start', 'start', 'start', 'start', 'start', 'start'};
symbol_names2 = {'body6', 'nose_a4', 'nose_h3', 'wing_at3', 'wing_ad4', 'wing_h6', 'tail_at3', 'tail_ad4', 'tail_h6'};
symbol_types2 = {'start', 'start', 'start', 'start', 'start', 'start', 'start', 'start', 'start'};

bin_distributions = struct;

for i=1:length(bin_id)
	bin_distributions(i).bin_id              = bin_id{i};
    bin_distributions(i).bin_needed          = zeros(1, m.params.T);
    bin_distributions(i).bin_nolonger_needed = zeros(1, m.params.T);
    
    for g = m.grammar.symbols
        
        if strcmp(g.name, symbol_names1{i})
            if strcmp('start', symbol_types1{i})
                bin_distributions(i).bin_needed = g.start_distribution;
            else
                bin_distributions(i).bin_needed = g.end_distribution;
            end
        end
        
        if strcmp(g.name, symbol_names2{i})
            if strcmp('start', symbol_types2{i})
                bin_distributions(i).bin_nolonger_needed = g.start_distribution;
            else
                bin_distributions(i).bin_nolonger_needed = g.end_distribution;
            end
            
        end
        
    end
end

end

