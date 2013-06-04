function [probs] = draw_normal(t, mumean, varmean, sigalpha, sigbeta)

timesig = sqrt(1/gamrnd(sigalpha, sigbeta));
timemu = normrnd(mumean, varmean);
probs = normpdf(t, timemu, timesig);
probs = probs / sum(probs);
