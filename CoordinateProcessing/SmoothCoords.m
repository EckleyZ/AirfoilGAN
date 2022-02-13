function out = SmoothCoords(Name,x,y)

%==========================================================================
%|||                         MODIFY COORDINATES                         |||
%==========================================================================
saveCoords = 1;

%intialize loop variables
pnum = 200;
angles = linspace(0,90,104);
spacing = 1-cosd(angles);
t_in = cell(7,1);

%pull in coordinates
p = [x, y];     %must be column vectors

%split into 7 segments
[~,MP] = min(p(:,1));
i = round(size(p,1)/14)-1;
seg = [round(linspace(1,MP-i,4)), round(linspace(MP+i,size(p,1),4))];


%find initial control points and smooth them
[p0mat,p1mat,p2mat,p3mat] = FindBzCP4AllSeg(p,seg);
[p1mat,p2mat] = BezierSmoothing(p0mat,p1mat,p2mat,p3mat,p);

%equally distribute points among segments based on cosine distribution
Enum = zeros(1,numel(seg)-1);
for E = [1 2 3 5 6 7]
    range = sort([p0mat(E,1), p3mat(E,1)]);
    valid = spacing(((spacing>range(1))+(spacing<=range(2)))==2);
    Enum(E) = sum((spacing>range(1))+(spacing<=range(2))==2);
    if E<4
        t_in{E} = flip(1-(valid-valid(1))*(1/(valid(end)-valid(1))));
    else
        t_in{E} = (valid-valid(1))*(1/(valid(end)-valid(1)));
    end

end
Enum(4) = pnum+7-sum(Enum);
Enum = Enum+1; %extra point at curve connection
if mod(Enum(4),2)==0 %even
    half = Enum(4)/2;
    tempSpacing = sind(angles(1:half));
    part1 = 0.5*flip(1-(tempSpacing/(tempSpacing(end))));
    part2 = 0.5*(spacing(1:half)/(spacing(half)))+0.5;
    t_in{4} = [part1(1:end-1), part2];
else
    P1 = (pnum/2)-sum(Enum(1:3))+7;
    P2 = (pnum/2)-sum(Enum(5:7))+7;
    part1 = 0.5*flip(1-spacing(1:P1)/spacing(P1));
    part2 = 0.5*(spacing(1:P2)/spacing(P2))+0.5;
    t_in{4} = [part1(1:end-1), part2];
end

%Create the curves
Q1 = bezierInterp(p0mat(1,:),p1mat(1,:),p2mat(1,:),p3mat(1,:),Enum(1),t_in{1});
Q2 = bezierInterp(p0mat(2,:),p1mat(2,:),p2mat(2,:),p3mat(2,:),Enum(2),t_in{2});
Q3 = bezierInterp(p0mat(3,:),p1mat(3,:),p2mat(3,:),p3mat(3,:),Enum(3),t_in{3});
Q4 = bezierInterp(p0mat(4,:),p1mat(4,:),p2mat(4,:),p3mat(4,:),Enum(4),t_in{4});
Q5 = bezierInterp(p0mat(5,:),p1mat(5,:),p2mat(5,:),p3mat(5,:),Enum(5),t_in{5});
Q6 = bezierInterp(p0mat(6,:),p1mat(6,:),p2mat(6,:),p3mat(6,:),Enum(6),t_in{6});
Q7 = bezierInterp(p0mat(7,:),p1mat(7,:),p2mat(7,:),p3mat(7,:),Enum(7),t_in{7});

%Combine Points
ModCoords = [Q1(1:end-1,:);Q2(1:end-1,:);Q3(1:end-1,:);Q4(1:end-1,:);...
             Q5(1:end-1,:);Q6(1:end-1,:);Q7(1:end-1,:)];

%create coordinate string and save to text file
if saveCoords==1
    saveFolder = 'ModifiedCoordinates\';
    txtStr = strcat(Name,sprintf('\n %0.5f   %0.5f',ModCoords'));
    fileID = fopen(strcat(saveFolder,Name,'.txt'),'w');
    fprintf(fileID,'%s',txtStr);
    fclose(fileID);
end

out = sprintf('Coordinates Modified for:   %s',Name);
