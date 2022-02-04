%Plot some data
clear;
close all;
clc;

%import a data file
Data = readtable("OutputPolars\Complete\avistar_Complete.txt");
keep = repmat(5:205,121,1)'+((0:120)*206);
Data = Data(keep(:),[1:4,9]);

%Break down by reynolds number
CL = zeros(201,121);
CD = zeros(201,121);
CM = zeros(201,121);
LD = zeros(201,121);
CP = zeros(201,121);
for t = 1:121
    start = 1+(t-1)*201;
    stop = t*201;
    CL(:,t) = Data{start:stop,2};
    CD(:,t) = Data{start:stop,3};
    CM(:,t) = Data{start:stop,4};
    LD(:,t) = Data{start:stop,5};
end

%plot them separtely
AOA = repmat((-25:0.25:25)',1,121);
Re = repmat((10:1.5:190),201,1);
fs = 14;

%reduce data density for plotting
aoa = AOA(2:2:end,2:2:end);
re = Re(2:2:end,2:2:end);
cl = CL(2:2:end,2:2:end);
cd = CD(2:2:end,2:2:end);
cm = CM(2:2:end,2:2:end);
ld = LD(2:2:end,2:2:end);

figure('InnerPosition',[50 50 1150 950]);
tiledlayout(2,2, 'Padding', 'none', 'TileSpacing', 'compact');
% subplot(2,2,1);
nexttile;
surf(aoa,re,cl);
xlabel('Angle of Attack [deg]','FontSize',fs);
ylabel('Re [1e5]','FontSize',fs);
zlabel('C_{L}','FontSize',fs);
title('Coefficient of Lift (C_{L})','FontSize',fs+6);
grid on
daspect([1 4 0.075]);
% set(get(gca,'xlabel'),'rotation',30);
% set(get(gca,'ylabel'),'rotation',-42);
% set(get(gca,'zlabel'),'rotation',90);

% subplot(2,2,2);
nexttile;
surf(aoa,re,cd);
xlabel('Angle of Attack [deg]','FontSize',fs);
ylabel('Re [1e5]','FontSize',fs);
zlabel('C_{D}','FontSize',fs);
title('Coefficient of Drag (C_{D})','FontSize',fs+6);
grid on
daspect([1 4 0.015]);
% set(get(gca,'xlabel'),'rotation',30);
% set(get(gca,'ylabel'),'rotation',-42);
% set(get(gca,'zlabel'),'rotation',90);

% subplot(2,2,3);
nexttile;
surf(aoa,re,cm);
xlabel('Angle of Attack [deg]','FontSize',fs);
ylabel('Re [1e5]','FontSize',fs);
zlabel('C_{M}','FontSize',fs);
title('Coefficient of Moment (C_{M})','FontSize',fs+6);
grid on
axis equal
daspect([1 4 0.005]);
% set(get(gca,'xlabel'),'rotation',30);
% set(get(gca,'ylabel'),'rotation',-42);
% set(get(gca,'zlabel'),'rotation',90);

% subplot(2,2,4);
nexttile;
surf(aoa,re,ld);
xlabel('Angle of Attack [deg]','FontSize',fs);
ylabel('Re [1e5]','FontSize',fs);
zlabel('L:D','FontSize',fs);
title('Lift to Drag Ratio','FontSize',fs+6);
grid on
axis equal
daspect([1 4 5]);
% set(get(gca,'xlabel'),'rotation',30);
% set(get(gca,'ylabel'),'rotation',-42);
% set(get(gca,'zlabel'),'rotation',90);



