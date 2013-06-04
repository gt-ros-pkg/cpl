figure(2)
clf
subplot(2,1,1)
plot(t,endprobs,'r')
hold on
plot(t,start1probs,'b')
plot(t,start2probs,'g')

endwaits = [];
for j = 1:endnum
    for i = 1:N
        constinteg = sum(-(dr+dd).^2.*endprobs(1:i,j)');
        %lininteg = sum(-(dr+dd).^2.*endprobs(i+1:round(i+(dr+dd)*rate),j)');
        lininteg = sum(-(max(dr+dd-(t(i+1:end)-t(i)),0)).^2.*endprobs(i+1:end,j)');
        start1integ = constinteg + lininteg;
        endwaits(j,i) = start1integ * sum(start1probs(i+1:end,j));
    end
end

startwaits = [];
for j = 1:start1num
    for i = 1:N
        startwaits(j,i) = sum(-(max(t(i)-t,0)).^2.*start2probs(:,j)');
    end
end

subplot(2,1,2)
%plot(t(1+dri:end),endwaits(1,1+dri:end),'r')
plot(t,endwaits,'r')
hold on
%plot(t(1+dri+ddi:end),startwaits(1,1+dri+ddi:end),'b')
plot(t,startwaits,'b')
minew = min(endwaits(:));
axis([0,Tend,1.4*minew,-0.2*minew])
