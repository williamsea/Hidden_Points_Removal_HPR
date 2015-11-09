% import x,y,z from cvs

figure
scatter3(x,y,z)

p = [x y z];
C = [0 0 100];
% visiblePtInds = HPR(p,C,pi);
param = 3;
dim=size(p,2);
numPts=size(p,1); %23751

p=p-repmat(C,[numPts 1]);%Move C to the origin
normp=sqrt(dot(p,p,2));%Calculate ||p||, dot product in , 23751*1
R=repmat(max(normp)*(10^param),[numPts 1]);%Sphere radius, 23751*1
P=p+2*repmat(R-normp,[1 dim]).*p./repmat(normp,[1 dim]);%Spherical flipping, 23751*3
visiblePtInds=unique(convhulln([P;zeros(1,dim)]));%convex hull, add one row [0 0 0], which is C moved to origin
visiblePtInds(visiblePtInds==numPts+1)=[]; % get rid of id==23752 point which is C (origin), total 11989 points

% p = [3.83488 -12.0954  -87.975]
% sqrt(dot(p,p,2))

figure 
visiblex = p(visiblePtInds,1);
visibley = p(visiblePtInds,2);
visiblez = p(visiblePtInds,3);
scatter3(visiblex,visibley,visiblez);

% figure
% trisurf(visiblePtInds, visiblex, visibley, visiblez)