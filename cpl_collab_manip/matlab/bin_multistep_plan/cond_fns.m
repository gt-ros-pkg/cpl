N = 1000;
A_mean = 8;
A_sig = 1;
t = linspace(0,15,N);
in_ws_A = ones(1,N);
%unavail_A = [14.99,15];
unavail_A = [6.5, 7;
             8, 10;
             11.5, 13];
prob_Ae = zeros(1,N);

last_mean = A_mean;
last_i = 1;
for i = 1:size(unavail_A,1)
    start_i = find(t>unavail_A(i,1),1);
    end_i = find(t>unavail_A(i,2),1);
    in_ws_A(t>unavail_A(i,1) & t<unavail_A(i,2)) = 0;
    prob_Ae(last_i:start_i) = normpdf(t(last_i:start_i),last_mean,A_sig);
    last_i = end_i;
    last_mean = last_mean + unavail_A(i,2) - unavail_A(i,1);
end
prob_Ae(last_i:end) = normpdf(t(last_i:end),last_mean,A_sig);

figure(1)
subplot(2,1,1)
plot(t,prob_Ae)
ylabel('P(t_e^A | t_s^A, E)')
subplot(2,1,2)
plot(t,in_ws_A)
axis([0,15,-0.1,1.1])
xlabel('Time (s)')
ylabel('InWorkspace(A)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


AB_mean = 8;
AB_sig = 1;
AB_mu = 1;
in_ws_B = ones(1,N);
%unavail_B = [15,15];
unavail_B = [6.5, 7;
             8, 10;
             11.5, 13];
prob_Bs = normpdf(t,AB_mean,AB_sig);
prob_Bs_orig = normpdf(t,AB_mean,AB_sig);

for i = 1:size(unavail_B,1)
    start_i = find(t>unavail_B(i,1),1);
    end_i = find(t>unavail_B(i,2),1);
    unavail_range = t>unavail_B(i,1) & t<unavail_B(i,2);
    removed_prob = sum(prob_Bs(unavail_range)) * (t(2) - t(1))
    in_ws_B(t>unavail_B(i,1) & t<unavail_B(i,2)) = 0;
    prob_Bs(t>unavail_B(i,1) & t<unavail_B(i,2)) = 0;
    len_t = N - end_i + 1;
    prob_Bs(end_i:end) = prob_Bs(end_i:end) + removed_prob*exppdf(t(1:len_t),AB_mu);
end

figure(2); clf
subplot(2,1,1)
plot(t,prob_Bs_orig,'-g')
ylabel('P(t_s^B | t_e^A, E)')
hold on
plot(t,prob_Bs)
subplot(2,1,2)
plot(t,in_ws_B)
axis([0,15,-0.1,1.1])
xlabel('Time (s)')
ylabel('InWorkspace(B)')
sum(prob_Bs) * (t(2) - t(1))
sum(prob_Bs_orig) * (t(2) - t(1))
