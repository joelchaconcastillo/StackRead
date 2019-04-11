function [ AUC, I, mina, specifityValues,sensitivityValues  ] = Segmentation( tophatFiltered, groundTruth )
sensitivityValues = zeros(256,1); % array to hold sensitivity values for different thresholds
specifityValues = zeros(256,1); % array to hold specifity values for different thresholds
%   Detailed explanation goes here
for thresh = 0:1:255  % loop for every threshold value
   I =imbinarize(tophatFiltered, thresh/255);   
    %counting negatives and positives    
    TP = sum(sum( groundTruth == 1 & I == 1));
    FP = sum(sum( groundTruth == 0 & I == 1));
    TN = sum(sum( groundTruth == 0 & I == 0));
    FN = sum(sum( groundTruth == 1 & I == 0));
    
    sensitivity = 1.0 * TP / (TP+FN); % calculate sensitivity
    specifity = 1.0 * FP / (FP+TN); % calculate specificty

    sensitivityValues(thresh+1) = sensitivity; %add to array
    specifityValues(thresh+1) = specifity; %add to array
end
%calculate nearest point to the left corner which is the best threshold value
%according to ROC
minv = 1;
for a= 1:256
    dist = sqrt((1-sensitivityValues(a))^2 + (0-specifityValues(a))^2);
    if(dist < minv)
        minv = dist;
        mina = a;
    end
end

%figure('Name','ROC','NumberTitle','off'),plot(specifityValues,sensitivityValues,'x');
%xlabel('FP Fraction');
%ylabel('TP Fraction');
X = [0;specifityValues;1];
Y = [0;sensitivityValues;1];
AUC = trapz(Y,X) 

I =imbinarize(tophatFiltered, mina/255);


I = bwareaopen(I,300);

I=bwpropfilt(I,'perimeter',1);
%I = bwconncomp(I);
end

