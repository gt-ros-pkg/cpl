function [B] = homo2d(x,y,r)

B = [cos(r), -sin(r), x;
     sin(r),  cos(r), y;
          0,       0, 1;];
