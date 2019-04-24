function [ ModelbyComponent ] = Modeling_B_Spline(Information_Components)
%MODELING_B_SPLINE Summary of this function goes here
%   Detailed explanation goes here
%%%%Select the branch points...
    IdComponents = unique(Information_Components(3,:));
    ModelbyComponent = [];
    for j =1:length(IdComponents) 
      listPoints = Information_Components( 1:2, Information_Components(3,:)==IdComponents(j));
      NPoints = length(listPoints(1,:));
      if NPoints <10
         continue;
      end
      %%Uniform knots method....
      k= 5;%min(5, NPoints);
      t = [repelem(0,k) linspace(0.0001,0.9, max(0,NPoints-k)*0.1) repelem(1,k)];
      D = bspline_estimate(k,t,listPoints);
       C = [D; repelem(IdComponents(j), length(D(1,:))); repelem(NPoints, length(D(1,:))) ];
   %   C = [bspline_deboor(k,t,D); repelem(NPoints, IdComponents(j)) ];
      ModelbyComponent = [ModelbyComponent  C];
    end
end

