function [ ModelbyComponent ] = ArteryModeling( skel, methodModeling)
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 
 %%splitting the full-skel through components..
  Information_Components = Components_Skel(skel);
  
 %%%%%%%%%%%
 %%%%%%%%%% extracting control points with the b-spline <----
 if methodModeling== 1
    ModelbyComponent = Modeling_B_Spline(Information_Components); 
 end
 %%%%%%%%%%%%%%%%
 %%%%%%%% extracting points with RDP algorithm...
 if methodModeling == 2
    ModelbyComponent = Modeling_RDP(Information_Components);
 end
 %%%%%%%%%%%%%
 %%%%%%%%% extracting points with eigenfeatures algorithm Jianbo Shi et al...
 if methodModeling == 3 
    ModelbyComponent = Modeling_CornerDetector(Information_Components, skel);
 end
end