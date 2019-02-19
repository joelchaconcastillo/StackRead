function [threshold] = Optimize(inImage)
 threshold=0.4;
 
 evaluateFunction(inImage, threshold); 
 
 figure(5)
for v = 1:100
    Y(v,1) =  evaluateFunction(inImage, v/100); 
    X(v,1) = v/100;
end
plot(X,Y)
xlabel('Threshold')
ylabel('Variance')
 %
 max(X)
 Y(max(X))
end    
    