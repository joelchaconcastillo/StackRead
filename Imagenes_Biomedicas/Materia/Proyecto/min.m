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

%imshow(skel)

%%%%Select the barnch points...
E = bwmorph(skel, 'branchpoints');
%%Structural elemen to split the branch points...

%%removing fake branchpoints..


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
  for k = 2:length(contour(:,1))
     try
        yy = spline(contour(indexstart:k,1), contour(indexstart:k,2), min(contour(indexstart:k,1)):max(contour(indexstart:k,1)))
     catch
         try
                     yy = spline(contour(indexstart:k,2), contour(indexstart:k,1), min(contour(indexstart:k,2)):max(contour(indexstart:k,2)))
         catch
             break;%indexstart=k;
         end
     end
  end

  Points_Skel = [ Points_Skel NewPoints];
  IDComponent = [IDComponent ones(1,length(NewPoints(1,:)))*j];
 % break
end

imshow(skel);
hold on;
plot(Points_Skel(2,:),Points_Skel(1,:),'bo')
[x y] = find(bwmorph(skel, 'branchpoints'));
Points_Skel = [transpose(x); transpose(y)];
plot(Points_Skel(2,:),Points_Skel(1,:),'ro')

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


