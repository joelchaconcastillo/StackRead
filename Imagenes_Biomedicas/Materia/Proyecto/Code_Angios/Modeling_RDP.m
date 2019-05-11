function [ ModelbyComponent ] = Modeling_RDP(Information_Components)
%MODELING_B_SPLINE Summary of this function goes here
%   Detailed explanation goes here
%%%%Select the branch points...
    IdComponents = unique(Information_Components(3,:));
    ModelbyComponent = [];
    for j =1:length(IdComponents) 
      listPoints = Information_Components( 1:2, Information_Components(3,:)==IdComponents(j));
      D = douglas_peucker( listPoints, 4);
      C = [D; repelem(IdComponents(j), length(D(1,:)))];
      ModelbyComponent = [ModelbyComponent  C];
    end
end

