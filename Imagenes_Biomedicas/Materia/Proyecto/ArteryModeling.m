function [ ModelbyComponent ] = ArteryModeling( skel)
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 
 %%splitting the full-skel through components..
  Information_Components = Components_Skel(skel)
  %skel2=zeros(size(skel));
  %skel2([Information_Components(2,:), Information_Components(1,:)])=1;
  %skel2(sub2ind(size(skel),Information_Components(1,:), Information_Components(2,:)))=1;
  %imshow(skel);
 % hold on;
%  plot(Information_Components(2,:), Information_Components(1,:),'bo')
  
 %%%%%%% extracting control points with the b-spline <----
 ModelbyComponent = Modeling_B_Spline(Information_Components); 
 %%% extracting points with RDP algorithm...
% ModelbyComponent = Modeling_RDP(Information_Components);
 %%% extracting points with eigenfeatures algorithm Jianbo Shi et al...
 %ModelbyComponent = Modeling_CornerDetector(Information_Components, skel);
 %%%%%%% extracting control points with the b-spline <----

  
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