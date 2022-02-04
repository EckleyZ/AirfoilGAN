% This function computes  euclidean distance  between p1 & p2
% p1 & p2 can be in N-dimension vector space
function Distance=TwoNormDist(p1,p2)
%INPUT
% p1=[x1,x2....xn) (or column vector)
% p2=[y1,y2....yn) (or column vector)
%OUTPUT
% Distance: 2-norm distance between p1 and p2

%% Not Vectorized
SquareDistance=0;

for i=1:length(p1)
      SquareDistance=SquareDistance + ( ( p1(i)-p2(i) ).^2 ) ;
end
Distance=sqrt(SquareDistance);


%%% Vectorized % I tested vectorized is slower !!!
% i=[1:length(p1)];
% Distance= norm(p1(i)-p2(i));



% Notes:
% Reference: http://en.wikipedia.org/wiki/Distance 