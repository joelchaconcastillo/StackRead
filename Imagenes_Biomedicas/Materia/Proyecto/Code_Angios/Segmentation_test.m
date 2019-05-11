function [I] = Segmentation_training( tophatFiltered, groundTruth )


I = (imbinarize(tophatFiltered,graythresh(tophatFiltered)));
%%%Connect small disconected elements...
st = strel('square', 5);
I = imdilate(I, st);
I = bwareaopen(I,400);
I=bwpropfilt(I,'perimeter',1);
end

