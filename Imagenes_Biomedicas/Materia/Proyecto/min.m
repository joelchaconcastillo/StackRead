
groundTruth  = imread('BD_20_Angios/1_gt.png');

tophatFiltered = Detection('BD_20_Angios/1.png');
[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );

skel = Skeletonization(I);

E = bwmorph(skel, 'branchpoints');
skel(E)=0;
components = bwconncomp(skel)
I=skel;

imPoints = I;
imPoints(:) = 0;
for i = 1:components.NumObjects
    tmp = zeros(size(I));
    tmp(components.PixelIdxList{i})=1;
   [r,c] = find(tmp);
   result = [result douglas_peucker([transpose(c);transpose(r)], 5)];
   %outImg = ArteryModeling(result, I);
   %imshow(outImg);
   %break;
end

%dilatedImage = imdilate(I,strel('disk',15));
%thinedImage = bwmorph(dilatedImage,'thin',inf);
%imPoints(sub2ind(size(I),result(2,:),result(1,:))) = 1;
outImg = ArteryModeling(result, I)


%imshow(imPoints)
%Vq = interp2(I,5);
%figure
%imagesc(Vq);
%colormap gray
%axis image
%axis off
%title('Linear Interpolation');