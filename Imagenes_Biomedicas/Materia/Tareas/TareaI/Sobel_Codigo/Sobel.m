clc;
    %% create a new figure to show the image . 
    newImg = imread('Waterlilies.jpg');
     %newImg = imread('me.jpg');
      newImg = imread('cells.jpg');
    
    %% show the loaded image.
     figure(1);
    imshow(newImg);
      
    %% convert RGB to gray scale.
    I = rgb2gray(newImg);
    I = im2double(I);
    %% based code sobel
    figure(2);
%    thres = Optimize(I); %get the best threshold..
    imshow(mySobel(I, 0.3));

    %%improved sobel
    figure(3);   imshow(edge(I,'sobel', Threshold));

    imshow(myImprovedSobel(I, 3));
    %%%%%
    %% library sobel
    figure(4);
    imshow(edge(I,'sobel', 0.04));
    
   
   %%%variance Otsu lanscapes...
  Threshold = Optimize(I); %% La idea es obtener el mejor threshold maximizando la varianza.. 
  
    
    
    