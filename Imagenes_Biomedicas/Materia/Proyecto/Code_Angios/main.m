%w = warning('query','last')
%id = w.identifier;
%warning('off',id)
addpath('../Bsplines');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PIPE PARAMS...
methodModeling = 1; %%%%%meaning flags---> 1: controlPoints of B-splines, 2:RDP algorithm, 3:Feature-selection algorithm.
SETIMAGE=2;%%%%%%%%%%%%%%%%5----> 1: set of training, 2: set of testing....
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if SETIMAGE ==1
    Bigskeleton = [];
    Bigskeletondeflated=[];
    BigTestDetected = [];
    BigGroundTruth = [];
    TotalPointsCompressed=0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training.......... 
    for i = 1:20
    groundTruth = imread(strcat(strcat('../BD_20_Angios/',num2str(i)),'_gt.png'));
    test = imread(strcat(strcat('../BD_20_Angios/',num2str(i)),'.png'));
    %%Enhances the contrast of the grayscale image I by transforming the values using contrast-limited adaptive histogram equalization
    test = adapthisteq(test);

    %%Detection ....
    testDetected = Detection(test);

    %Improving and normalizing image....
    testDetected = uint8(round(255*(testDetected-min(testDetected(:)))/(max(testDetected(:)) - min(testDetected(:)))));
    BigTestDetected = [BigTestDetected testDetected];
    BigGroundTruth = [BigGroundTruth groundTruth];

    [AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation_training( testDetected, groundTruth/255 );

    %%%%%%SKELETONIZATION...
    skel = Skeletonization(I);
    Bigskeleton = [Bigskeleton skel];
    %%%%%%%MODELING...

    ModelbyComponent = ArteryModeling(skel, methodModeling);
    TotalPointsCompressed = TotalPointsCompressed + length(ModelbyComponent(1,:));
    %%%%%%INTERPOLATING SKELETON...
    new_skel = Sampling_Model(test, ModelbyComponent, methodModeling);
    %imshow(new_skel);
    %return;
    Bigskeletondeflated = [Bigskeletondeflated new_skel];
    %%Getting components....
    %plot(ModelbyComponent(2,:), ModelbyComponent(1,:),'ro')
    end
    %%%%%%%%%%%%% AUC, corr, jaccardDistance, compressRatio..



    %%%JACCAR DISTANCE.... no suggested....
    intersectImg = Bigskeletondeflated & Bigskeleton; 
    unionImg = Bigskeletondeflated | Bigskeleton;
    numerator = sum(intersectImg(:));
    denomenator = sum(unionImg(:));
    jaccardIndex = numerator/denomenator;
    jaccardDistance = 1 - jaccardIndex;

    %%%%%%CORRELATION....
    corr2(Bigskeleton, Bigskeletondeflated)

    %%%%%%%%%%%%%%%%AUC of all images...
    [AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation_training( BigTestDetected, BigGroundTruth/255 );
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%TEST!!!!!!!!!

if SETIMAGE == 2
    Bigskeleton = [];
    Bigskeletondeflated=[];
    BigTestDetected = [];
    BigGroundTruth = [];
    TotalPointsCompressed=0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing.......... 
    for i = 21:30
        groundTruth = imread(strcat(strcat('../Coronarias_test/',num2str(i)),'_gt.png'));
        test = imread(strcat(strcat('../Coronarias_test/',num2str(i)),'.png'));
        

        %%Enhances the contrast of the grayscale image I by transforming the values using contrast-limited adaptive histogram equalization
        test = adapthisteq(test);

        %%Detection ....
        testDetected = Detection(test);

        %Improving and normalizing image....
        testDetected = uint8(round(255*(testDetected-min(testDetected(:)))/(max(testDetected(:)) - min(testDetected(:)))));

        BigTestDetected = [BigTestDetected testDetected];
        BigGroundTruth = [BigGroundTruth groundTruth/255];
        I = Segmentation_test(testDetected);
        %%%%%%SKELETONIZATION...
        skel = Skeletonization(I);
        Bigskeleton = [Bigskeleton skel];
        %%%%%%%MODELING...
        ModelbyComponent = ArteryModeling(skel, methodModeling);
        %TotalPointsCompressed = TotalPointsCompressed + length(ModelbyComponent(1,:));
        %%%%%%INTERPOLATING SKELETON...
        new_skel = Sampling_Model(test, ModelbyComponent, methodModeling);

        Bigskeletondeflated = [Bigskeletondeflated new_skel];
        %%Getting components....
%        imshow(imfuse(skel,new_skel));
 %       hold on;
%        plot(ModelbyComponent(2,:), ModelbyComponent(1,:),'ro')
        

    end
    %%%%%%%%%%%%% AUC, corr, jaccardDistance, compressRatio..


    %%%JACCAR DISTANCE.... no suggested....
    intersectImg = Bigskeletondeflated & Bigskeleton; 
    unionImg = Bigskeletondeflated | Bigskeleton;
    numerator = sum(intersectImg(:));
    denomenator = sum(unionImg(:));
    jaccardIndex = numerator/denomenator;
    jaccardDistance = 1 - jaccardIndex

    %%%%%%CORRELATION....
    corr2(Bigskeleton, Bigskeletondeflated)

    %%%%%%%%%%%%%%%%AUC of all images...
    [AUC, I, mina, specifityValues,sensitivityValues]  = Segmentation_training( BigTestDetected, BigGroundTruth );

    %%%compress ratio...
    sum(Bigskeleton(:)>0)/TotalPointsCompressed;
end
