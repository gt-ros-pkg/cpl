function [Bs] = get_row_slots(N, w_t, w_b, row_center)
% N = number of slots in row, w_t = table width, w_b = bin width, w_m = margin width

w_m = (w_t - N*w_b) / (N+1);
for i = 1:N
    y_b_i = 0.5*w_t - (i*w_m + (i-0.5)*w_b);
    Bs{i} = row_center * homo2d(0.0, y_b_i, 0.0);
end
