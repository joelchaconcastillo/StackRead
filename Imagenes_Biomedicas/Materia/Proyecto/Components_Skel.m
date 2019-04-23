function [ output_args ] = Components_Skel( skel )
%COMPONENTS_SKEL Summary of this function goes here
%   Detailed explanation goes here

%%Structural elemen to split the branch points...
E = bwmorph(skel, 'branchpoints');

%%%structural element used in the disconection
st = strel('square', 3);
skelDivided = zeros(size(skel));
skelDivided(E) = 1;
%%getting the divided components....
%removed = 
[Components, NComponents]= bwlabeln(skel.*imcomplement(imdilate(skelDivided, st)));

Points_Skel = [];
IDComponent=[];
EndPointsbyComponent=[];
ControlPoints = [];
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
      EndPointsbyComponent = [EndPointsbyComponent ; contour(1,:), contour(length(contour(:,1)),:)];

      contour = unique(contour,'rows','stable');
    %  imshow(SingleCurve);
      %NewPoints = douglas_peucker(transpose(contour), 2);
      %fitting each component....
      %%fit(contour(:,1), contour(:,2),'poly2')

    %  t = [repelem(0,k) linspace(0.0001,0.9,length(contour(:,1))*0.1) repelem(1,k)];
      tic
      if length(contour(:,1)) > 4
      D = detectMinEigenFeatures(SingleComponent);
      D = transpose(D.Location);
      D = NewPoints;%[D(2,:); D(1,:)];
      %D = bspline_estimate(k,t,transpose(contour));

      end
      Points_Skel = [ Points_Skel NewPoints];
      IDComponent = [IDComponent ones(1,length(NewPoints(1,:)))*j];
       j
    end


end

