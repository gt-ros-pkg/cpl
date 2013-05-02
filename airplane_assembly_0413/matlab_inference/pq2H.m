
function H = pq2H( pq )

%     p = pq(1:3);
%     q = pq(4:7);
% 
% 
%     if abs(1-norm(q)) > 0.01
%         H2 = [];
%     else
% 
%         q = Quaternion(q);
%         H2 = q.T;
%         H2(1:3,4) = p';
%     end
    
    if norm(pq) < 0.9
        H = [];
    else
    
        H          = eye(4);
        H(1:3,4)   = pq(1:3)';
        H(1:3,1:3) = quat2dcm(pq(4:7)')';
    end
%     disp(pq');
%     assert(norm(H - H2) < 10e-3);
end