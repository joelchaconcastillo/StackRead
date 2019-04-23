function [ outImg ] = ArteryModeling( skel)
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 
 %%splitting the full-skel through components..
  = Components_Skel(skel); 
 %%%%%%% extracting control points with the b-spline <----
 %%%%%%% time-consuming...
 Modeling_B_Spline(skel);
 
 
end

