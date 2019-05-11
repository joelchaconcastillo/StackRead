%w = warning('query','last')
%id = w.identifier;
addpath('../Bsplines');
%warning('off',id) 
c = jet(7);
methodModeling = 1; %%%%%meaning flags---> 1: controlPoints of B-splines, 2:RDP algorithm, 3:Feature-selection algorithm.

Bigskeleton = [];
Bigskeletondeflated=[];
BigTestDetected = [];
BigModelbyComponent = [];
TotalPointsCompressed=0;
antskel=[];
cont = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training.......... 

for i = 43:47
    imshow(zeros(size(test)))
    hold on;
    test = imread(strcat(strcat('../ProyParcialTSIB/Secuencia/',num2str(i)),'.png'));
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
   
  % if cont > 1
  %  imshow(imfuse(skel,antskel));
  % else
  %  imshow(skel);
  % end
  %  hold on all;
  %  IdComponents = unique(ModelbyComponent(3,:)); 
  %  for j =1:length(IdComponents) 
 %     listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
   
%     plot(listControlPoints (2,:), listControlPoints (1,:), 'o', 'MarkerSize', 4, 'Color', c(1,:));
%    text(double(listControlPoints(2,1))+10, double(listControlPoints(1,1))+10,strcat(' ',num2str(j)), 'fontsize',18, 'color', 'red');
   % end

  if cont > 0
    corr2(skel, antskel)
   end
    antskel = skel;
    Bigskeleton = [Bigskeleton skel];
    BigTestDetected = [BigTestDetected testDetected];
    ModelbyComponent = [ModelbyComponent; repelem(cont+1, length(ModelbyComponent(1,:)))];
    BigModelbyComponent = [BigModelbyComponent ModelbyComponent];
    row = length(BigModelbyComponent(:,1));
    IdComponents = unique(BigModelbyComponent(row,:)); 
    for j =1:length(IdComponents) 
      listControlPoints = BigModelbyComponent( 1:2, BigModelbyComponent(row,:)==IdComponents(j));
   

%    text(double(listControlPoints(2,1))+10, double(listControlPoints(1,1))+10,strcat(' ',num2str(j)), 'fontsize',18, 'color', 'red');
     plot(listControlPoints (2,:), listControlPoints (1,:), 'o', 'MarkerSize', 4, 'Color', c(j+1,:));
    end
   
   % pause(3);
    cont = cont+1;
end
%%%%%%%%%%%%%%%%Checking the correlation bewteen frames....
 row = length(BigModelbyComponent(:,1));
 IdComponents = unique(BigModelbyComponent(row,:)); 
    for j =1:(length(IdComponents)-1) 
        img1 = zeros(size(test));
        img2 = zeros(size(test));
      listControlPoints1 = BigModelbyComponent( 1:2, BigModelbyComponent(row,:)==IdComponents(j));
      listControlPoints2 = BigModelbyComponent( 1:2, BigModelbyComponent(row,:)==IdComponents(j+1));
      Points1 = [];
       for  i =1:length(listControlPoints1(1,:))
        if listControlPoints1(1,i) < 1 || listControlPoints1(2,i) < 1
           continue ;
        end
        Points1 = [Points1  listControlPoints1(:,i)];
       end
        Points2 = [];
       for  i =1:length(listControlPoints2(1,:))
        if listControlPoints2(1,i) < 1 || listControlPoints2(2,i) < 1
           continue ;
        end
        Points2 = [Points2  listControlPoints2(:,i)];
       end
       img1(sub2ind(size(test),round(Points1(1,:)), round(Points1(2,:))))=1;
       img2(sub2ind(size(test),round(Points2(1,:)), round(Points2(2,:))))=1;
     
       corr2(img1, img2)
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
