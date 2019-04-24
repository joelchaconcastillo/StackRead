addpath('Bsplines');

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

%%%%%%SKELETONIZATION...
skel = Skeletonization(I);

%%%%%%%MODELING...
ModelbyComponent = ArteryModeling(skel);
new_skel = Sampling_Model(test, ModelbyComponent);
imshow(new_skel);
%%Modeling artery......
%splined = [splined ArteryModeling(result, I)];


%outImg(sub2ind(size(I),uint8(result(1,:)),uint8(result(2,:)))) = 1
%imshow(skel);
%hold on;
%plot(Points_Skel(2,:),Points_Skel(1,:),'ro')
%return;

break
end

% return
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


