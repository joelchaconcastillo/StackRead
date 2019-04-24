function [ ModelbyComponent ] = Modeling_CornerDetector(Information_Components, skel)
%MODELING_B_SPLINE Summary of this function goes here
%   Detailed explanation goes here
%%%%Select the branch points...
    IdComponents = unique(Information_Components(3,:));
    ModelbyComponent = [];
    for j =1:length(IdComponents) 
      listPoints = Information_Components( 1:2, Information_Components(3,:)==IdComponents(j));
      tmpI = zeros(size(skel));
       tmpI(sub2ind(size(skel),round(listPoints(2,:)), round(listPoints(1,:))))=1;  
      D = detectMinEigenFeatures(tmpI);
      D = transpose(D.Location);
%min      imshow(tmpI);
      C = [D; repelem(IdComponents(j), length(D(1,:)))];
      ModelbyComponent = [ModelbyComponent  C];
    end
end

