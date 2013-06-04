function [probs] = gen_data(distribs, rate, t_duration)

N = t_duration*rate+1;
numbins = size(distribs,1);

t = linspace(0,t_duration,N);

% beginning and ending probabilities for the bins steps
probs = cell(numbins,2);
cur_mean = 0;
for i = 1:numbins
    cur_mean = cur_mean + distribs(i,1);
    probs{i,1} = normpdf(t, cur_mean, distribs(i,2));
    probs{i,1} = probs{i,1} / sum(probs{i,1}) * distribs(i,5);
    cur_mean = cur_mean + distribs(i,3);
    probs{i,2} = normpdf(t, cur_mean, distribs(i,4));
    probs{i,2} = probs{i,2} / sum(probs{i,2}) * distribs(i,5);
end

if 0
    figure(1)
    clf
    hold on
    for i = 1:numbins
        plot(t,probs{i,1},'r')
        plot(t,probs{i,2},'b')
    end
end
