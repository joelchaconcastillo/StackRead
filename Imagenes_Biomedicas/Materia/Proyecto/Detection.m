function [ tophatFiltered ] = Detection(Img)
%STAGE_1 Summary of this function goes here
%   Detailed explanation goes here
%Img = imread(filename_Image);
%se = strel('diamond',[19, 19]);
se = strel('disk',19)

tophatFiltered = imtophat(255-Img,se); 
end

