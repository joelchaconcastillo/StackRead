w = warning('query','last')
id = w.identifier;
addpath('Bsplines');
warning('off',id) 
c = jet(75);
methodModeling = 1; %%%%%meaning flags---> 1: controlPoints of B-splines, 2:RDP algorithm, 3:Feature-selection algorithm.

Bigskeleton = [];
Bigskeletondeflated=[];
BigTestDetected = [];

TotalPointsCompressed=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training.......... 
for i = 43:47
    test = imread(strcat(strcat('ProyParcialTSIB/Secuencia/',num2str(i)),'.png'));
    test = rgb2gray(test);
    %%Enhances the contrast of the grayscale image I by transforming the values using contrast-limited adaptive histogram equalization
%    test = adapthisteq(test);
 
    %%Detection ....
    testDetected = Detection(test);
    %Improving and normalizing image....
    testDetected = uint8(round(255*(testDetected-min(testDetected(:)))/(max(testDetected(:)) - min(testDetected(:)))));
    I = Segmentation_test(testDetected);
    %%%%%%SKELETONIZATION...
    skel = Skeletonization(I);    
    %%%%%%%MODELING...
    ModelbyComponent = ArteryModeling(skel, methodModeling);
    TotalPointsCompressed = TotalPointsCompressed + length(ModelbyComponent(1,:));
    %%Getting components....
    imshow(skel);  
    hold on all;
    IdComponents = unique(ModelbyComponent(3,:)); 
    for j =1:length(IdComponents) 
      listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
    %plot(listControlPoints (2,:), listControlPoints (1,:), 'Color',colorstring(5))        
     plot(listControlPoints (2,:), listControlPoints (1,:), 'o', 'MarkerSize', 4, 'Color', c(j,:));
    text(double(listControlPoints(2,1))+10, double(listControlPoints(1,1))+10,strcat(' ',num2str(j)), 'fontsize',18, 'color', 'red');
    end
    
    pause(1);
    Bigskeleton = [Bigskeleton skel];
    BigTestDetected = [BigTestDetected testDetected];
end
%%%%%%%%%%%%% AUC, corr, jaccardDistance, compressRatio..


%%%JACCAR DISTANCE.... no suggested....
%intersectImg = Bigskeletondeflated & Bigskeleton; 
%unionImg = Bigskeletondeflated | Bigskeleton;
%numerator = sum(intersectImg(:));
%denomenator = sum(unionImg(:));
%jaccardIndex = numerator/denomenator;
%jaccardDistance = 1 - jaccardIndex;

%%%%%%CORRELATION....
%corr2(Bigskeleton, Bigskeletondeflated)

%%%compress ratio...
%sum(Bigskeleton(:)>0)/TotalPointsCompressed;
