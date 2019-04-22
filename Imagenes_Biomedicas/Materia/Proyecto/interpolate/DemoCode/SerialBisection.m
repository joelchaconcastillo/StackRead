%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name: SerialBisection.m
% Description: This function employes parallel method to divide input data
% into small single-b-spline pieces having the maximum fitted error being
% smaller than a certain input value
% Prototype: 
% VectorUX = ParallelBisection(DataIn,Degree,CtrlError,NumOfPieceStart)
% Input parameters:
% - DataIn: A matrix contains input data, having at least 2 columns. 1st
% column is parametric, 2nd column is X, 3rd column is Y and so on.
% - Degree: Degree of the fitted B-spline
% - CtrlError: Maximum fitted error of a single piece b-spline
% Output parameters:
% - VectorUX: a matrix contains information of each single piece b-spline.
% each column stores the information of each single piece b-spline.
% 1st row: start index, 2nd: end index, 3 to 2+Order^2: Coefficient vector,
% 3 + Order^2 to end - 1: B-spline control points, end: maximum fitted
% error.
% Version: 1.0                                                             
% Date: 30-June-2016
% Author: Dvthan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function VectorUX = SerialBisection(DataIn,Degree,CtrlError)
%% Public variables
% prepare Amat
dataT = DataIn(:,1);
[datasize, S] = size(DataIn);
% Amat = [1,x1,...,x1^N ;1,x2,...,x2^N ;...];
Amat = zeros(datasize,Degree+1);
Amat(:,1) = 1;
for ii = 1:(Degree)
    Amat(:,1+ii) = Amat(:,ii).*dataT;
end
Ymat = DataIn(:,2:end);

%% Bisecting
StartIdx = 1;
EndIdx = datasize;  
ptr = 1;
% one piece B-spline fitting
% computing Nmatrix
Order = Degree + 1;
Order2= Order*Order;
VectorUX1 = zeros(3+Order*(S-1)+Order*Order, fix(datasize/Order));
% Calculation of Ni,0 format [a0,a1,...,an-1,an]

while EndIdx > StartIdx  
        
    LeftIdx = StartIdx;
    RightIdx = EndIdx;    
    Knotin(1)= dataT(StartIdx);
    while(1)  
        if (StartIdx + Degree)>=datasize
            CtrlPointYsave = zeros(Order,(S-1));
            Nmatrixsave = zeros(Order2,1);
            ErrorMaxsave  = 0;
            break;
        end
        Knotin(2)= dataT(EndIdx);
        % Basis function calculation    
        SPNmat = NmatCal_1P(Knotin,Degree);
       %Spline fitting
        clear Nmatrix  Ymatrix
        Nmatrix = Amat(StartIdx:EndIdx,:)*reshape(SPNmat,[Order,Order]);
        Ymatrix = Ymat(StartIdx:EndIdx,:);
        CtrlP = Nmatrix\Ymatrix;        
        
        R = Ymatrix-Nmatrix*CtrlP;
        R = R.*R;
        Errorcal = max(sum(R,2));    
        Errorcal = sqrt(Errorcal);
        % Fitting error evaluation

        success  = 0;
        if Errorcal <= CtrlError
            success = 1;
            CtrlPointYsave = CtrlP;
            Nmatrixsave = SPNmat;
            ErrorMaxsave  = Errorcal;                    
        end
        if success ==1            
            LeftIdx = EndIdx;            
        else
            RightIdx = EndIdx;
        end
        if RightIdx - LeftIdx <=1
            break;
        end
        EndIdx = floor((LeftIdx+RightIdx)/2);
    end
    % fill vector Ux 
    VectorUX1(1, ptr) = StartIdx;
    VectorUX1(2, ptr) = LeftIdx;
    VectorUX1(3:(2+Order*Order), ptr) = Nmatrixsave';
    VectorUX1(3+Order2:end-1,ptr) = reshape(CtrlPointYsave,[Order*(S-1),1]);    
    VectorUX1(end, ptr) = ErrorMaxsave;
    ptr = ptr + 1;
    StartIdx = LeftIdx+1;
    EndIdx = datasize;
end
VectorUX = VectorUX1(:,1:(ptr-1));
end
%% 
%% Dependent function
function Nmat = NmatCal_1P(Knotin,Degree)
coder.inline('always');
Order = Degree + 1;
rownumbers = (Order)*(sum(1:(Order)));
CoeNmat = zeros(rownumbers,1);
CoeNmat(1) = 1;
Knot = ones(1,2*Order);
Knot(1:Order)=Knotin(1);
Knot(Order+1:end)=Knotin(2);
for jj=1:Degree %each degree or Ni,j
      id0 = Order*sum(0:(jj-1))+1;
      id1 = Order*sum(0:(jj))+1;
      for kk = 0:jj %each id member matrix at iterval ii
         id2 = Order-jj+kk; %effective internal we focus,eg when ii=4,jj=1,then id 2=3,4
         id2Knot00 = id2 + jj; % effective knot num 1
         id2Knot01 = id2Knot00 + 1;
         if (id2>0)&&(id2Knot01<=numel(Knot));
             % Access previous data Ni-1,j-1 Ni,j-1 and Ni+1,j-1
             id00 = id0 + (kk-1)*Order;
             id01 = id0+kk*Order;
             if kk==0 %first box of matrix
                 N0 = zeros(Order,1);
                 N1 = CoeNmat(id01:(id01+Degree));
             elseif kk==(jj)
                 N0 = CoeNmat(id00:id00+Degree);
                 N1 = zeros(Order,1);
             else
                 N0= CoeNmat(id00:id00+Degree);
                 N1= CoeNmat(id01:(id01+Degree));                    
             end

               % calculate a1x+a0, 
              aden = (Knot(id2Knot00)-Knot(id2));
              bden = (Knot(id2Knot01)-Knot(id2+1));
              a0 = 0;
              a1 = 0;
             if aden~= 0
             a1 = 1/aden;
             a0 = -Knot(id2)/aden; 
             end
             % calculate b1x+b0%
             b0 = 0;
             b1 = 0;
             if  bden~= 0
             b1 = -1/bden;
             b0 = Knot(id2Knot01)/bden; 
             end
           % Multiplication, 
             Acoef = zeros(Order,1);
             N00 = a0*N0;
             N01 = a1*N0;
             N10 = b0*N1;
             N11 = b1*N1;
             Acoef(1)=N00(1)+N10(1);
             for n=2:Order 
                 Acoef(n) = N00(n)+N10(n)+ N01(n-1)+N11(n-1);
             end
             id11=id1+kk*Order;
             CoeNmat(id11:id11+Degree)=Acoef;
          end
      end      
end
id10 = Order*sum(0:(Degree))+1;
Nmat = CoeNmat(id10:end);
end