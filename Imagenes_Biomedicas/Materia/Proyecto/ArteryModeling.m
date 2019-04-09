function [ outImg ] = ArteryModeling( bImg, I )
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 %%Dividing the points to get a good interpolation
 outImg =zeros(size(I));
 xq = min(bImg(2,:)):max(bImg(2,:));
 yq = pchip(bImg(2,:),bImg(1,:),xq);
 outImg(sub2ind(size(I),uint8(xq),uint8(yq))) = 1;
end

