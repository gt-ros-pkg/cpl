
T0 = 100;
T1 = 50;

r = ones(1, T1);

while sum(r) < T0,
    i = randi([1, 10]);
    r(i) = r(i) + 1;
end

%% test down & up 1
a0 = nxmakegaussian(T0, 20, 1)/2 + nxmakegaussian(T0, 50, 5)/2;
a1 = down_sample_vrts(a0, r);
a10 = up_sample_vrts(a1, r);

plot(a0);
hold on;
plot(a10, '.r');
hold off



%% test down & up 2

b0 = nxmakegaussian(T0, 20, 1)/2 + nxmakegaussian(T0, 30, 1)/2;
for i=2:T0
    b0(i,i:end) = b0(1, 1:end-i+1);
end

b1  = down_sample_vrts_mat(b0, r);
b10 = up_sample_vrts_mat(b1, r);

subplot(1, 3, 1); imagesc(b0);
subplot(1, 3, 2); imagesc(b1);
subplot(1, 3, 3); imagesc(b10);

%% test 3

c0 = a0 * b0;
c1 = a1 * b1;
c10 = up_sample_vrts(c1, r);

plot(c0)

hold on;
plot(c10, 'r');

hold off







