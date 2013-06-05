function mod = quatmod( q )
%  QUATMOD Calculate the modulus of a quaternion.
%   N = QUATMOD( Q ) calculates the modulus, N, for a given quaternion, Q.  
%   Input Q is an M-by-4 matrix containing M quaternions.  N returns a 
%   column vector of M moduli.  Each element of Q must be a real number.  
%   Additionally, Q has its scalar number as the first column.
%
%   Examples:
%
%   Determine the modulus of q = [1 0 0 0]:
%      mod = quatmod([1 0 0 0])
%
%   See also QUATCONJ, QUATDIVIDE, QUATINV, QUATMULTIPLY, QUATNORM,
%   QUATNORMALIZE, QUATROTATE.

%   Copyright 2000-2005 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2005/11/01 23:39:32 $

mod = sqrt(quatnorm( q ));
