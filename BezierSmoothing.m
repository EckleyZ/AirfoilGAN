function [p1mat,p2mat] = BezierSmoothing(p0mat,p1mat,p2mat,p3mat,p)
%{
    BEZIER CURVE SMOOTHING
    =======================================================================
    This script works to join multiple segments of Bezier Curves so that
    the joints between segments are seamless. Modifies the control points
    of 2 separate cubic Bezier curves to help them join smoothly.

    Inputs:
        p0mat = Nx2 matrix with the x and y coordiantes of the start point
                for every bezier curve
        p1mat = Nx2 matrix with the x and y coordinates of the first
                control point for each bezier curve
        p2mat = Nx2 matrix with the x and y coordinates of the second
                control point for each bezier curve
        p3mat = Nx2 matrix with the x and y coordiantes for the end point
                of every bezier curve
    
    Outputs:
        p1mat = Nx2 matrix with the modified coordinates of the first
                control point for every curve
        p1mat = Nx2 matrix with the modified coordinates of the second
                control point for every curve

%}

%initialize new points
np1 = zeros(size(p0mat,1)-1,2);
np2 = zeros(size(p0mat,1)-1,2);
% figure();
% fill(p(:,1),p(:,2),[0.5 0.5 0.5]);
% hold on

for m = 1:size(p0mat,1)-1
    %pull from p2mat and p1mat
    c1 = p2mat(m,:);    %control point at end of first curve
    c2 = p1mat(m+1,:);  %control point at start of second curve
    c = p3mat(m,:);     %connection point
    
    %find distances (d1 is first curve, d2 is second curve)
    d1 = norm(c1-c);
    d2 = norm(c2-c);
    %matching point for both sides
    m1 = d1/d2*(c-c2)+c;    %matches with first curve
    m2 = d2/d1*(c-c1)+c;    %matches with second curve
    
    
    t1 = atan2d((c1(2)-c(2)),(c1(1)-c(1)));
    t2 = atan2d(m1(2)-c(2),m1(1)-c(1));
    t = t2-t1;
    if abs(t)>180
        if t2<0
            t2 = t2+360;
        else
            t1 = t1+360;
        end
        t = t2-t1;
    end
    
    %Rotate points based on distance ratio
    rt = d2/(d1+d2)*t;
    np1(m,:) = ([cosd(rt), -sind(rt); sind(rt), cosd(rt)]*(c1-c)')'+c;
    np2(m,:) = ([cosd(rt), -sind(rt); sind(rt), cosd(rt)]*(m2-c)')'+c;
    
    %plots for debugging
    
%     plot([c1(1),c(1)],[c1(2),c(2)],'ro-','MarkerFaceColor','r','linewidth',3);    %real curve 1
%     plot([c2(1),c(1)],[c2(2),c(2)],'bo--','MarkerFaceColor','b','linewidth',3);   %real curve 2
%     plot([m1(1),c(1)],[m1(2),c(2)],'ro--','MarkerFaceColor','r','linewidth',3);   %colinear with curve 2
%     plot([m2(1),c(1)],[m2(2),c(2)],'bo-','MarkerFaceColor','b','linewidth',3);    %colinear with curve 1
%     scatter(c(1),c(2),70,'k','filled'); %rotation point
%     scatter(np1(m,1),np1(m,2),50,'r^','markerfacecolor','k','linewidth',3); %Settled point 1
%     scatter(np2(m,1),np2(m,2),50,'b^','markerfacecolor','k','linewidth',3); %settled point 2
%     set(gca,'linewidth',2);
%     axis equal;
%     grid on
    
    
    
end
%replace original control points with new points
p2mat(1:m,:) = np1;
p1mat(2:m+1,:) = np2;