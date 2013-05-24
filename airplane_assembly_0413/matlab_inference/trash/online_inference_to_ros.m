
clc;
clear;

load d4;

om = m;

%% connect to ros app

connt = tcpip('localhost', 12342);
connt.OutputBufferSize = 8 * 1000;
fopen(connt);


%%


for ttt=1:1000
    
    pause(0.001)
    
    % read point & update
    while connt.BytesAvailable > 10 + 8 * 3

        name  = fread(connt, 10, 'char');
        name  = char(name');
        point = fread(connt, 3, 'double');
        
        disp(name);
        disp(point');
        
    end
    
    for i=1:length(m.g)
        if m.g(i).isterminal
            m.g(i).likelihood = om.g(i).likelihood;
            m.g(i).likelihood(ttt+1:end,ttt+1:end) = exp(m.g(i).log_null_likelihood);
            m.g(i).likelihood = triu(m.g(i).likelihood);
        end
    end

    % check enough new data & find t
    do_inference = 0;
    if 1
        do_inference = 1;
    end

    % inference
    if do_inference
        tic
        m = m_inference(m);
        dataset.grammar.symbols = calculate_symbol_distribution(m, dataset.grammar.symbols);
        toc
    end

    % send inference data to ros app
    if do_inference
        for i=1:length(dataset.grammar.symbols)

            s = dataset.grammar.symbols(i);

            if 1

                name = [s.name '_start'];
                while length(name) < 10, name = [name ' ']; end;
                fwrite(connt, name, 'char');
                fwrite(connt, s.start_distribution, 'double');

                name = [s.name '_end'];
                while length(name) < 10, name = [name ' ']; end;
                fwrite(connt, name, 'char');
                fwrite(connt, s.end_distribution, 'double');

            end


            if s.name == 'S'

                plot(s.end_distribution);

            end
        end

    end
        
end

fclose(connt)



