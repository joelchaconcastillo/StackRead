function [ out] = Skeletonization( Image )
%SKELETONIZATION Summary of this function goes here
%   Detailed explanation goes here

%%I =bwmorph(Image,'skel',Inf);
%%dilatedImage = imdilate(I,strel('disk',11));
%%skel = bwmorph(dilatedImage,'thin',inf);
% % B = bwmorph(skel, 'branchpoints');
% % E = bwmorph(skel, 'endpoints');
% % [y,x] = find(E);
% % B_loc = find(B);
% % Dmask = false(size(skel));
% % for k = 1:numel(x)
% %     D = bwdistgeodesic(skel,x(k),y(k));
% %     distanceToBranchPt = min(D(B_loc));
% %     Dmask(D < distanceToBranchPt) =true;
% % end
% % skelD = skel - Dmask;
% % imshow(skelD);
% % hold all;
% % [y,x] = find(B); plot(x,y,'ro')
% % 
% % 
%%%%%%%%%%%%%55

%dilatedImage = imdilate(Image,strel('disk',3));
%Image = bwmorph(dilatedImage,'thin',inf);
out =bwmorph(Image,'skel',Inf);
%out =bwmorph(Image,'thin',Inf);
%out =bwmorph(out,'skel',Inf);

%dilatedImage = imdilate(I,strel('disk',8));
%out =bwmorph(dilatedImage,'thin',inf);

return
%Start off with your code above then do the following

%I got a better starting image with the 'thin' option than the 'skel' option
I = bwmorph(out,'thin',Inf);

%Alternative splitting method to 'branchpoint'
%Use convolution to identify points with more than 2 neighboring pixels
filter = [1 1 1;
          1 0 1;
          1 1 1];

I_disconnect = I & ~(I & conv2(double(I), filter, 'same')>2);

cc = bwconncomp(I_disconnect);
numPixels = cellfun(@numel,cc.PixelIdxList);
[sorted_px, ind] = sort(numPixels);

%Remove components shorter than threshold
threshold  = 15;
for ii=ind(sorted_px<threshold)
    cur_comp = cc.PixelIdxList{ii};
    I(cur_comp) = 0; 

    %Before removing component, check whether image is still connected
    full_cc = bwconncomp(I);
    if full_cc.NumObjects>1
        I(cur_comp) = 1; 
    end
end
%Clean up left over spurs
out = bwmorph(I, 'spur');

end

