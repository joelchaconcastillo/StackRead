function [ ModelbyComponent ] = ArteryModeling( skel)
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 
 %%splitting the full-skel through components..
  Information_Components = Components_Skel(skel); 
 %%%%%%% extracting control points with the b-spline <----
 %%%%%%% time-consuming...
 ModelbyComponent = Modeling_B_Spline(Information_Components); 
%imshow(skel);
%hold on;
%plot(Points_Skel(2,:),Points_Skel(1,:),'bo')
%plot(ControlPoints(2,:),ControlPoints(1,:),'yo')
%[x y] = find(bwmorph(skel, 'branchpoints'));
%Points_Skel = [transpose(x); transpose(y)];
%plot(Points_Skel(2,:),Points_Skel(1,:),'ro')

%points = detectSURFFeatures(skel)
%plot(points.Location(:,1),points.Location(:,2),'ro')
%return
%points = detectMinEigenFeatures(skel)
%plot(points.Location(:,1),points.Location(:,2),'ro')

end