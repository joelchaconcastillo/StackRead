
groundTruth  = imread('/home/user/StackRead/Imagenes_Biomedicas/Materia/Proyecto/BD_20_Angios/1_gt.png');

tophatFiltered = Detection('/home/user/StackRead/Imagenes_Biomedicas/Materia/Proyecto/BD_20_Angios/1.png');
[AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation( tophatFiltered, groundTruth/255 );
I = Skeletonization(I);
%imshow(I);
pause(4)
[r,c] = find(I);
result = douglas_peucker([transpose(c);transpose(r)], 1);
I(:)=0;
I(sub2ind(size(I),result(2,:),result(1,:))) = 1;
imshow(I);