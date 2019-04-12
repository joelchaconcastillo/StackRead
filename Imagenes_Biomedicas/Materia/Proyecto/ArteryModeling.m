function [ outImg ] = ArteryModeling( Points, I )
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 %%Dividing the points to get a good interpolation
 outImg =zeros(size(I));
fitcknn(Points(1,:),Y,'NumNeighbors',5,'Standardize',1)
 outImg(sub2ind(size(I),uint8(Points(1,:)),uint8(Points(2,:)))) = 1;
 imshow(outImg)
 
 

return
 
 %%this procedure is for each component of the set of points...
  Setgenereal =  unique(transpose(Points),'rows');
    Partition =[];
for i = 1:length(Setgenereal(:,1))
  Set = Setgenereal;
 % [x,i] = min(Set(:,2));
  %i=2
  p1 = Set(i,:);
  Set(i,:) = [];
  [i, d] = dsearchn(Set, p1);
  p2 = Set(i,:);
  Set(i,:) = [];

 while length(Set(:,1))>0
     [i, d] = dsearchn(Set, p2); 
     p3 = Set(i,:);
     Set(i,:) = [];
     if  d > 100
         continue
     end
      n1 = (p2 - p1) / norm(p2 - p1); 
      n2 = (p2 - p3) / norm(p2 - p3);  % Normalized vectors
      angle =  (180/pi)*atan2(norm(det([n2; n1])), dot(n1, n2));
      if angle < 90
           subset = [ p1;p2;p3];
          if ((p2(1) < p1(1) && p2(1) < p3(1)) || (p2(1) > p1(1) && p2(1) > p3(1)) )&& (p1(2) ~= p2(2)) && (p3(2) ~= p2(2) ) && (p3(2) ~= p1(2) ) 
            xq =  min(subset(:,2)):max(subset(:,2));
            yq = pchip(subset(:,2),subset(:,1),xq);
            Partition = [Partition; yq' xq'];
          elseif ((p2(2) < p1(2) && p2(2) < p3(2)) || (p2(2) > p1(2) && p2(2) > p3(2)) )&& (p1(1) ~= p2(1)) && (p3(1) ~= p2(1) ) && (p3(1) ~= p1(1) )
             xq =  min(subset(:,1)):max(subset(:,1));
             yq = pchip(subset(:,1),subset(:,2),xq);
             Partition = [Partition; xq' yq'];
          end
      end
     p1=p2;
     p2=p3;
 end

end
outImg(sub2ind(size(I),uint8(Partition(:,2)),uint8(Partition(:,1)))) = 1;
 dilatedImage = imdilate(outImg,strel('disk',10));
outImg = bwmorph(dilatedImage,'thin',inf);
 %imshow(outImg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imshow(I)

%  %initialization...
%  angle = 0;
%   
%  Set =  unique(transpose(Points),'rows');
%  Partition = [];
% 
%  %pick up the right-bottom point
%  [x,i] = min(Set(:,2));
%  i=10
%  p1 = Set(i,:);
%  Set(i,:) = [];
% 
%  [i, d] = dsearchn(Set, p1);
%  p2 = Set(i,:);
%  Set(i,:) = []; 
% 
% Partition = [p1;p2];
%  n1 = (p2 - p1) / norm(p2 - p1); 
% 
%  
% [i, d] = dsearchn(Set, p2); 
% p3 = Set(i,:);
% Set(i,:) = [];
% n2 = (p2 - p3) / norm(p2 - p3);  % Normalized vectors
% angle =  (180/pi)*atan2(norm(det([n2; n1])), dot(n1, n2))
% 
%  while angle < 90
%      Partition = [Partition;p3];
%      p2=p3;
%      [i, d] = dsearchn(Set, p2);
%      p3 = Set(i,:);
%      Set(i,:) = [];
%      n2 = (p2 - p3) / norm(p2 - p3);  % Normalized vectors
%      angle = (180/pi)*atan2(norm(det([n2; n1])), dot(n1, n2))%atan2(norm(det([n2; n1])), dot(n1, n2));%(180/pi)*atan2(abs(det([p3-p1;p2-p1])),dot(p3-p1,p2-p1));%acos(dot(n1, n2));  % atan2(norm(det([n2; n1])), dot(n1, n2))
%  %    (180/pi)*acos(dot(n1, n2))   
%      if length(Set(:,1)) == 0
%         break 
%      end
%  end
%  
% 
%  [coeff,score,latent,tsquared,explained,mu]  = pca(Partition)
%  outImg(sub2ind(size(I),Partition(:,2),Partition(:,1))) = 1;
%  imshow(outImg)
%  Partition = (Partition)/coeff
% % Partition = Partition + abs(min(Partition))+1
% % outImg(sub2ind(size(I),uint8(Partition(:,2)),uint8(Partition(:,1)))) = 1;
%  %imshow(outImg)
%  
%  xq = min(Partition(:,1)):max(Partition(:,1));
%  yq = pchip(Partition(:,2),Partition(:,1),xq);
%  Partition = [xq; yq];
%  Partition = abs((Partition')*coeff')
%  %outImg(sub2ind(size(I),uint8(Partition(:,1)),uint8(Partition(:,2)))) = 1;
%  %imshow(outImg)
end

