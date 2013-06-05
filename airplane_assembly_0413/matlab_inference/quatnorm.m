function qout = quatnorm( q )
%  QUATNORM Calculate the norm of a quaternion.
%   N = QUATNORM( Q ) calculates the norm, N, for a given quaternion, Q.  Input
%   Q is an M-by-4 matrix containing M quaternions.  N returns a column vector
%   of M norms.  Each element of Q must be a real number.  Additionally, Q has
%   its scalar number as the first column.
%
%   Examples:
%
%   Determine the norm of q = [1 0 0 0]:
%      norm = quatnorm([1 0 0 0])
%
%   See also QUATCONJ, QUATDIVIDE, QUATINV, QUATMOD, QUATMULTIPLY, 
%   QUATNORMALIZE, QUATROTATE.

%   Copyright 2000-2005 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2005/11/01 23:39:34 $

if any(~isreal(q(:)))
    error('aero:quatnorm:isnotreal','Input elements are not real.');
end

if (size(q,2) ~= 4)
    error('aero:quatnorm:wrongdim','Input dimension is not M-by-4.');
end

qout = sum(q.^2,2);

