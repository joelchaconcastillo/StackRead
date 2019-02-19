function [Variance] = evaluateFunction(inImage, threshold)
   %%computig the variance of the image..
    binaryimage = mySobel(inImage, threshold);
    binaryimage = (binaryimage == 255);
    %binaryimage = edge(inImage,'sobel', threshold)
    
    Omega0 = sum(inImage(binaryimage==0));
    Omega1 = sum(inImage(binaryimage==1));
    Mu0 = mean(inImage(binaryimage == 0));
    Mu1 = mean(inImage(binaryimage == 1));
    Variance = sqrt(Omega0*Omega1*((Mu0-Mu1)^2));
end