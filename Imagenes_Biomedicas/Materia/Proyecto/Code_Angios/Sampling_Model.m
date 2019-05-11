function [ new_skel ] = Sampling_Model( test,  ModelbyComponent, methodModeling)
%SAMPLING_MODEL Summary of this function goes here
%   Detailed explanation goes here

%%%%%  B_Spline sampling.....
if methodModeling == 1
 IdComponents = unique(ModelbyComponent(3,:));
 new_skel = zeros(size(test));
 
    for j =1:length(IdComponents) 
      listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
      %%Uniform knots method....
      NPoints = ModelbyComponent( 4, ModelbyComponent(3,:)==IdComponents(j));
      NPoints = NPoints(1);
     k= min(4, NPoints);
     t = [repelem(0,k) linspace(0.0001,0.9, round(max(0,(NPoints-k)*0.0))) repelem(1,k)];
      D =  listControlPoints ;
      C = (bspline_deboor(k,t,D));
      %%%adjusting the out image points...
      Points = [];
      for  i =1:length(C(1,:))
        if C(1,i) < 1 || C(2,i) < 1
           continue ;
        end
        Points = [Points  C(:,i)];
      end
      new_skel(sub2ind(size(test),round(Points(1,:)), round(Points(2,:))))=1;      
    end
end
    
%%%%% RDP sampling...
if methodModeling == 2 
     IdComponents = unique(ModelbyComponent(3,:));
     new_skel = zeros(size(test)); 
        for j =1:length(IdComponents) 
          listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
          %%Uniform knots method....
          NPoints = length(listControlPoints(1,:)); 
          k= min(6, NPoints);
          t = [repelem(0,k) linspace(0.0001,0.9, round(max(0,(NPoints-k))))   repelem(1,k)];
          C = (bspline_deboor(k,t,listControlPoints));
          new_skel(sub2ind(size(test),round(C(1,:)), round(C(2,:))))=1;      
        end
end


%%%% corners sampling...
if methodModeling == 3
 IdComponents = unique(ModelbyComponent(3,:));
 new_skel = zeros(size(test)); 
    for j =1:length(IdComponents) 
      listControlPoints = ModelbyComponent( 1:2, ModelbyComponent(3,:)==IdComponents(j));
      %%Uniform knots method....
      NPoints = length(listControlPoints(1,:)); 
      k= min(6, NPoints);
      t = [repelem(0,k) linspace(0.0001,0.9, round(max(0,(NPoints-k))))   repelem(1,k)];
      C = (bspline_deboor(k,t,listControlPoints));
      new_skel(sub2ind(size(test),round(C(1,:)), round(C(2,:))))=1;      
    end
end
