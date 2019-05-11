groundTruth  =[];
test =[];
% for i = 1:20 %%Training....
% groundTruth = [groundTruth imread(strcat(strcat('BD_20_Angios/',num2str(i)),'_gt.png'))];
% test = [test imread(strcat(strcat('BD_20_Angios/',num2str(i)),'.png'))];
% %test = [test imread('BD_20_Angios/1.png')];
%end
%imshow(test)
%return 
skeleton = [];
splined= [];
for i = 12:20
groundTruth = imread(strcat(strcat('BD_20_Angios/',num2str(i)),'_gt.png'));
test = imread(strcat(strcat('BD_20_Angios/',num2str(i)),'.png'));
%test = [test imread('BD_20_Angios/1.png')];
%se = strel('diamond',10);
%test=imgaussfilt(test, 4);
imshow(test);
%https://la.mathworks.com/help/vision/ref/matchfeatures.html

%return
tophatFiltered = Detection(test);

%tophatFiltered = adapthisteq(tophatFiltered);
%tophatFiltered = uint8(tophatFiltered)+uint8(255-test);
%tophatFiltered = adapthisteq(tophatFiltered);
%tophatFiltered = Detection(255-tophatFiltered);


%tophatFiltered = tophatFiltered - min(tophatFiltered(:));
%se = strel('square',3)
%tophatFiltered = imclose(tophatFiltered,se);
%imshow(tophatFiltered);
%return;
%tophatFiltered = adapthisteq(tophatFiltered);

imshow(tophatFiltered);


%imshow(tophatFiltered)
% return;

[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );
%se = strel('square',25);
%tophatFiltered  = imbothat(test,se);
%AUC2, I2, mina2, specifityValues2,sensitivityValues2]  = Segmentation( tophatFiltered, groundTruth/255 );
%tophatFiltered = 0.5*uint8(255-test)+0.5*uint8(I);
%[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );
%I=imgaussfilt(255*I, 1);
%imshow(I);
%return

skel = Skeletonization(I);
hold all;


[i j] = find(bwmorph(skel, 'end'));

maxdistance = -inf;
indexa = -1;
indexb = -1;
bestPath = skel;
for a = 1:numel(i)
    for b = (a+1):numel(i)
        D1 = bwdistgeodesic(skel,j(a),i(a), 'quasi-euclidean');
        D2 = bwdistgeodesic(skel,j(b),i(b), 'quasi-euclidean');
        D = D1 + D2;
        D = round(D * 8) / 8;
        D(isnan(D)) = inf;
        paths = imregionalmin(D);
        if maxdistance < sum(paths(:))
            maxdistance = sum(paths(:));
            indexa=a;
            indexb=b;
            bestPath=paths;
        end
    end
end
imshow(bestPath);
hold on;


   contour=[];
   try
     contour = bwtraceboundary(bestPath,[i(indexa),j(indexa)],'N', 8, Inf);
  catch
      contour = bwtraceboundary(bestPath,[i(indexa),j(indexa)],'N', 8, Inf, 'counterclockwise');
  end
   contour = unique(contour,'rows','stable');

   k = 3;
  %t = [repelem(0,k) repelem(1,k)];
  t = [repelem(0,k) linspace(0.0001,0.9,20) repelem(1,k)];  
   tic;
  D = bspline_estimate(k,t,transpose(contour));
  toc
  C = bspline_deboor(k,t,D);
   
   
plot(C(2,:),C(1,:),'bo')




%for n = 1:numel(i)
%    text(j(n),i(n),[num2str(D(i(n),j(n)))],'color','g');
%end

return;
%%%%Select the barnch points...
E = bwmorph(skel, 'branchpoints');

%%Structural elemen to split the branch points...



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

for j =1:NComponents
    %j=91
    if sum(Components == j) < 5
        continue;
    end
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
  NewPoints = douglas_peucker(transpose(contour), 4);
  %fitting each component....
  %%fit(contour(:,1), contour(:,2),'poly2')
  indexstart=1;
  k = 3;
    t = [repelem(0,k) repelem(1,k)];
  %t = [0 0 0 linspace(0.0001,0.9,3) 1 1 1];
  
 % D = bspline_estimate(k,t,transpose(contour));
 % C = bspline_deboor(k,t,D);
 % NewPoints = C;
  %for k = 2:length(contour(:,1))
%                yy = spline(contour(indexstart:k,1), contour(indexstart:k,2), min(contour(indexstart:k,1)):max(contour(indexstart:k,1)))
   %  try
   %     yy = spline(contour(indexstart:k,1), contour(indexstart:k,2), min(contour(indexstart:k,1)):max(contour(indexstart:k,1)))
   %  catch
   %      try
   %                  yy = spline(contour(indexstart:k,2), contour(indexstart:k,1), min(contour(indexstart:k,2)):max(contour(indexstart:k,2)))
   %      catch
   %          break;%indexstart=k;
   %      end
   %  end
 % end

  Points_Skel = [ Points_Skel NewPoints];
  IDComponent = [IDComponent ones(1,length(NewPoints(1,:)))*j];
  break
  j
end


imshow(skel);
hold on;
plot(Points_Skel(2,:),Points_Skel(1,:),'bo')
%[x y] = find(bwmorph(skel, 'branchpoints'));
%Points_Skel = [transpose(x); transpose(y)];
%plot(Points_Skel(2,:),Points_Skel(1,:),'ro')

%points = detectSURFFeatures(skel)
%plot(points.Location(:,1),points.Location(:,2),'ro')
%return
%points = detectMinEigenFeatures(skel)
%plot(points.Location(:,1),points.Location(:,2),'ro')

%%Modeling artery......
%splined = [splined ArteryModeling(result, I)];


%outImg(sub2ind(size(I),uint8(result(1,:)),uint8(result(2,:)))) = 1
%imshow(skel);
%hold on;
%plot(Points_Skel(2,:),Points_Skel(1,:),'ro')
%return;

break
end

 return
intersectImg = splined & skeleton; 
unionImg = splined | skeleton;
numerator = sum(intersectImg(:));
denomenator = sum(unionImg(:));
jaccardIndex = numerator/denomenator
jaccardDistance = 1 - jaccardIndex

%dilatedImage = imdilate(I,strel('disk',15));
%thinedImage = bwmorph(dilatedImage,'thin',inf);
%imPoints(sub2ind(size(I),result(2,:),result(1,:))) = 1;
%outImg = ArteryModeling(result, I)
%imshow(imPoints)


