function [outImage] = mySobel(inImage, threshold)
   Gx = [1 2 1; 0 0 0; -1 -2 -1]; Gy = Gx';
   diff_x = conv2(double(inImage), Gx, 'same');
   diff_y = conv2(double(inImage), Gy, 'same');
   tmp = sqrt(diff_x.*diff_x + diff_y.*diff_y);
   outImage = uint8((tmp> threshold) * 255);
end