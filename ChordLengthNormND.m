% Normalized Chord-length parameterization, domain is b/w 0 and 1 inclusive

% INPUT
% Data: A set of n points (p1,p2,...pn) , in N-dimension vector space (i.e. points to be approximated by Cubic Bezier Curve
%       e.g in 2D p1=(x1,y1); in 3D p1=(x1,y1,z1) ; in 4D p1=(w1,x1,y1,z1)
%       Each row of p contains values of a point
%       e.g.p for 2D p=[5 0;    % p1
%                       1 2;    % p2
%                       3 4;    % p3
%                       4 0]    % p4

% OUTPUT
% paramterized values of t=[t1,t2,...,tn], (normalized b/w 0 to 1)

function [t]=ChordLengthNormND(p)

n=size(p,1); % number of rows in p
TotalDistance(1:n-1)=0; dSum=0;
for i=1:n-1
    dSum=dSum+TwoNormDist( p(i+1,:), p(i,:) );
    TotalDistance(i)=dSum; % chord-length of ith segment (length upto ith segment)
end

% when dSum=0 (all points have same value.)
if(dSum==0) 
    t(1:n-1)=0;
    t(n)=1;
    return
end

% % Normalizing (Vectorized)

t(1)=0;
i=[2:n-1];
t(i)=TotalDistance(i-1)/dSum;           
t(n)=1;



% % % Normalizing (Not Vectorized)
% t(1)=0;
% for i=2:n-1
%     t(i)=TotalDistance(i-1)/dSum;            
% end
% t(n)=1;
