function string = nxstr_rmdup(string, rlen)
%NXSTR_RMDUP Summary of this function goes here
%   Detailed explanation goes here

if ~exist('rlen')
    rlen = 1;
end;

for i=length(string):-1:2
%     if string(i) == string(i-1)
%         string(i) = [];
%     end
    
    if i+rlen-1 <= length(string) && i-rlen >= 1
        if strcmp(string(i:i+rlen-1), string(i-rlen:i-1))
            string(i:i+rlen-1) = [];
        end
    end
end

end

