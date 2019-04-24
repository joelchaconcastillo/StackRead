function [ new_skel ] = Sampling_Model( test,  ModelbyComponent)
%SAMPLING_MODEL Summary of this function goes here
%   Detailed explanation goes here

%%%%%  B_Spline sampling.....

 IdComponents = unique(ModelbyComponent(3,:));
 new_skel = zeros(size(test));
    for j =1:length(IdComponents) 
      listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
      %%Uniform knots method....
      NPoints = ModelbyComponent( 4, ModelbyComponent(3,:)==IdComponents(j));
      NPoints = NPoints(1);
      k= min(5, NPoints);
      t = [repelem(0,k) linspace(0.0001,0.9, max(0,NPoints-k)*0.1) repelem(1,k)];
      D =  listControlPoints ;
      C = uint8(bspline_deboor(k,t,D));
      new_skel(sub2ind(size(test),C(1,:), C(2,:)))=1;
    end
end

