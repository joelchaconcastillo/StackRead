function [ out] = Skeletonization( Image )
%SKELETONIZATION Summary of this function goes here
%   Detailed explanation goes here
out =bwmorph(Image,'skel',Inf);

end

