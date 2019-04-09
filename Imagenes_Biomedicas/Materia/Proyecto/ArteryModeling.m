function [ outImg ] = ArteryModeling( Points, I )
%ARTERYMODELING Summary of this function goes here
%   Detailed explanation goes here
 %find()
 %%Dividing the points to get a good interpolation
 outImg =zeros(size(I));
 
 %%this procedure is for each component of the set of points...
 

 %initialization...
 angle = 0;
  
 Set =  unique(transpose(Points),'rows');
 Partition = [];

 %pick up the right-bottom point
 [x,i] = min(Set(:,2));
 p1 = Set(i,:);
 Set(i,:) = [];

 [i, d] = dsearchn(Set, p1);
 p2 = Set(i,:);
 Set(i,:) = []; 

 [i, d] = dsearchn(Set, p2);
 p3 = Set(i,:);
 Set(i,:) = [];

Partition = [p1;p2];
 n2 = (p2 - p1) / norm(p2 - p1); 
 while angle < 90
     Partition = [Partition;p3];
     n1 = (p3 - p1) / norm(p3 - p1);  % Normalized vectors
        
     angle = (180/pi)*acos(dot(n1, n2));%atan2(norm(det([n2; n1])), dot(n1, n2));%(180/pi)*atan2(abs(det([p3-p1;p2-p1])),dot(p3-p1,p2-p1));%acos(dot(n1, n2));  % atan2(norm(det([n2; n1])), dot(n1, n2))
     (180/pi)*acos(dot(n1, n2))
     angle
    % p1 = p2;
     p2 = p3;
     [i, d] = dsearchn(Set, p2);
     p3 = Set(i,:);
     Set(i,:) = [];
     if length(Set(:,1)) == 0
        break 
     end
 end
 
 outImg(sub2ind(size(I),Partition(:,2),Partition(:,1))) = 1;
 imshow(outImg)
 xq = min(Partition(:,1)):max(Partition(:,1));
 yq = pchip(Partition(:,1),Partition(:,2),xq);
outImg(sub2ind(size(I),uint8(xq),uint8(yq))) = 1;
end

