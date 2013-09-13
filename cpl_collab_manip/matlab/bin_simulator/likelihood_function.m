function [d] = likelihood_function(dists, nowtimeind, likelihood_params)
sigma         = likelihood_params.sigma;
latent_noise  = likelihood_params.latent_noise;
future_weight = likelihood_params.future_weight;
d = zeros(size(dists));
d(:,1:nowtimeind) = (exp(-dists(:,1:nowtimeind).^2/(2.0*sigma^2)) + latent_noise) / future_weight;
d(:,nowtimeind:end) = 1.0;
d(isnan(d)) = 0.0;
