start = 1;
t = linspace(0,1000/rate,1000);
figure(243)
subplot(2,2,start)
hold on
plot(nowtimesec*[1 1], [0, 1.1], 'k-')
plot(t,probs{1,1},'r')
plot(t,probs{2,1},'b')
plot(t,probs{3,1},'g')
xlim([0 60])
ylim([0 0.12])
subplot(2,2,start+1)
hold on
plot(nowtimesec*[1 1], [0, 2.1], 'k-')
plot(t,probs{1,1},'r')
plot(t,probs{4,1},'m')
plot(t,probs{5,1},'k')
xlim([0 60])
ylim([0 0.12])
