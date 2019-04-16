groundTruth  =[];
test =[];
% for i = 1:20 %%Training....
% groundTruth = [groundTruth imread(strcat(strcat('BD_20_Angios/',num2str(i)),'_gt.png'))];
% test = [test imread(strcat(strcat('BD_20_Angios/',num2str(i)),'.png'))];
% %test = [test imread('BD_20_Angios/1.png')];
%end
%imshow(test);
%return 
skeleton = [];
splined= [];
for i = 1:20
groundTruth = imread(strcat(strcat('BD_20_Angios/',num2str(i)),'_gt.png'));
test = imread(strcat(strcat('BD_20_Angios/',num2str(i)),'.png'));
%test = [test imread('BD_20_Angios/1.png')];

tophatFiltered = Detection(test);

[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );


skel = Skeletonization(I);

E = bwmorph(skel, 'branchpoints');
skel(E)=0;
%E = bwmorph(skel, 'endpoints');

%hold all;
%[y,x] = find(E);
%plot(x,y,'ro')
%return

components = bwconncomp(skel);
I=skel;
skeleton = [skeleton skel];
imPoints = I;
imPoints(:) = 0;
result = [];
for i = 1:components.NumObjects
    tmp = zeros(size(I));
    tmp(components.PixelIdxList{i})=1;
   [r,c] = find(tmp);
   if length(r)< 1
       continue
    end
   result = [result dpsimplify([transpose(r);transpose(c)], 1)];
end
outImg = zeros(size(I));
outImg(sub2ind(size(I),uint8(result(1,:)),uint8(result(2,:)))) = 1;
 imshow(outImg)
%splined = [splined ArteryModeling(result, I)];

break
end
 
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


