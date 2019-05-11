function [ tophatFiltered ] = Detection(Img)
%STAGE_1 Summary of this function goes here
%   Detailed explanation goes here

%se = strel('square',29);
%tophatFiltered = imtophat(255-Img,se);
%return;
%Img = 255-Img;

%%%%%%%%%%%%%%%%%%%%%%%%%FRAMES Parameters
L=65;
T=65;
Sigma=2.4;
k=12;

%%%%%%%%%%%%%%%%%%%%%% Overall Experiment 
% L=15;
% T=15;
% Sigma=2.4;
% k=12;
theta = 0:k:180; %different rotations
out = zeros(size(Img))-1000000;
Reject = (Img(:) < 0); 

m = sqrt((T*T)+(L*L))/2 +1;%max((T-1)/2,(L-1)/2); %%it need to be a square or the convolved image will be smaller
[x,y] = meshgrid(-m:m,-m:m); % non-rotated coordinate system, contains (0,0)
for t = theta
   t = t / 180 * pi;      % angle in radian
   u = cos(t)*x - sin(t)*y; % rotated coordinate system
   v = sin(t)*x + cos(t)*y; % rotated coordinate system
   N = (abs(u) <= (T-1)/2 )& (abs(v) <= L/2); % domain
   k = -exp(-(u.^2)/(2*Sigma*Sigma)); % kernel
 %  k = k - min(0, min(k(:)))
 %ma = mean(k(N))
   k = k -  mean(k(N));
   k(~N) = 0;%0.03;%ma+0.01;%mean(k(N))+0.01;       
  % set kernel outside of domain to 0
   res = conv2(double(Img),k,'same');
   out = max(out,res);
   %imshow(k);
   %break
end
%imshow(out/max(out(:)))
out = out/max(out(:)); % force output to be in [0,1] interval that MATLAB likes
out(Reject) = 0;
tophatFiltered=out;

%%GMF.....

end

