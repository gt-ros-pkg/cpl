function [h] = relevance_heur(t, startprobs, endprobs, binprob, nowtimeind, endedweight, notbranchweight)

probended = sum(endprobs(nowtimeind:end));
expectedstart = sum(startprobs .* t);
h = (t(nowtimeind)-expectedstart) + endedweight*(1-1/probended) + notbranchweight*(1-1/binprob);
