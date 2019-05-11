function [ Information_Components ] = Components_Skel( skel )
%COMPONENTS_SKEL Summary of this function goes here
%   Detailed explanation goes here

%%Structural elemen to split the branch points...
E = bwmorph(skel, 'branchpoints');

%%%structural element used in the disconection
st = strel('square', 3);
skelDivided = zeros(size(skel));
skelDivided(E) = 1;

[Components, NComponents]= bwlabeln(skel.*imcomplement(imdilate(skelDivided, st)));
Information_Components = [];
  for j =1:NComponents
      SingleComponent = zeros(size(skel));
      SingleComponent = bwmorph(Components==j, 'endpoints');
      SingleComponent = logical(skel.*imdilate(SingleComponent,st)+(Components==j)+skelDivided);
      SingleComponent = bwareafilt(SingleComponent,1);
      %SingleComponent = bwmorph(SingleComponent, 'endpoints');
      [x, y]=find(SingleComponent.*skelDivided,1);  
      %getting line...
      try
         contour = bwtraceboundary(SingleComponent,[x y],'N', 8, Inf);
      catch
          contour = bwtraceboundary(SingleComponent,[x y],'N', 8, Inf, 'counterclockwise');
      end
      contour = transpose(unique(contour,'rows','stable'));
      contour = [contour; repelem(j , length(contour(1,:))) ];
      Information_Components = [Information_Components  contour];
  end
end

