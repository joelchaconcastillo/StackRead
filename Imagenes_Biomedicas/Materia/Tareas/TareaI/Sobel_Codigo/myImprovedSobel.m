function [outImage] = mySobel2(inImage, threshold)
   Gx = [2 3 0 -3 -2; 3 4 0 -4 -3; 6 6 0 -6 -6; 3 4 0 -4 -3; 2 3 0 -3 -2]; Gy = Gx';
   G45 = [0 -2 -3 -2 -6; 2 0 -4 -6 -2; 3 4 0 -4 -3; 2 6 4 0 -2; 6 2 3 2 0]; 
   G135 = [-6 -2 -3 -2 0; -2 -6 -4 0 2; -3 -4 0 4 2; -2 0 4 6 2; 0 2 3 2 6];
   diff_x = conv2(double(inImage), Gx, 'same');
   diff_y = conv2(double(inImage), Gy, 'same');
   diff45 = conv2(double(inImage), G45, 'same');
   diff135 = conv2(double(inImage), G135, 'same');
 % outImage = imgradient(inImage, 'sobel');
   tmp = sqrt(diff_x.*diff_x + diff_y.*diff_y + diff45.*diff45 +diff135.*diff135 );
   outImage = uint8((tmp > threshold) * 255);
end