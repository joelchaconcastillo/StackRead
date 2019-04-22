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

tophatFiltered = Detection(test);
tophatFiltered = tophatFiltered - min(tophatFiltered(:));
%imshow(tophatFiltered);
%return;
%tophatFiltered = adapthisteq(tophatFiltered);
%imshow(tophatFiltered)
% return;

[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );
%se = strel('square',25);
%tophatFiltered  = imbothat(test,se);
%AUC2, I2, mina2, specifityValues2,sensitivityValues2]  = Segmentation( tophatFiltered, groundTruth/255 );
imshow(I);
%return;
skel = Skeletonization(I);

hold all;

%imshow(skel)

%%%%Select the barnch points...
E = bwmorph(skel, 'branchpoints');
%%Structural elemen to split the branch points...

st = strel('square', 3);
skelDivided = zeros(size(skel));

skelDivided(E) = 1;

%%getting the divided components....
%removed = 
[Components, NComponents]= bwlabeln(skel.*imcomplement(imdilate(skelDivided, st)));

Points_Skel = [];
IDComponent=[];
for j =1:NComponents
  SingleComponent = zeros(size(skel));
  SingleComponent(find(bwmorph(Components==j, 'endpoints')))=1;
  SingleComponent = logical(skel.*imdilate(SingleComponent,st)+(Components==j)+skelDivided);
  SingleComponent = bwareafilt(SingleComponent,1).*skelDivided ;
  [x, y]=find(SingleComponent,1);
  %finding coordinates...
  contour = bwtraceboundary(SingleCurve,[x y],'N');
  contour = unique(contour,'rows','stable');
%  imshow(SingleCurve);
  NewPoints = douglas_peucker(transpose(contour), 4);
  Points_Skel = [ Points_Skel NewPoints];
  IDComponent = [IDComponent ones(1,length(NewPoints(1,:)))*j]
end
imshow(skel);
hold on;
%[x y] = find(bwmorph(skel, 'branchpoints'));
%Points_Skel = [transpose(x); transpose(y)];
plot(Points_Skel(2,:),Points_Skel(1,:),'ro')
points = detectMinEigenFeatures(skel)
%plot(points.Location(:,1),points.Location(:,2),'bo')
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


