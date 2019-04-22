
function [Nmat] = NewNmatrix(Knot,Degree)
coder.inline('always');
%Build up matrix format and definition.
Order = Degree + 1;
Intervals = numel(Knot)-1;
rownumbers = (Order)*(sum(1:(Order)));
CoeNmat = zeros(rownumbers,Intervals);
activeKnot = zeros(1,Intervals);
% Caculate Ni,0 for o degree
for ii = 1 : Intervals
    if (Knot(ii+1) - Knot(ii)) ~=0
        CoeNmat(1,ii) = 1;
        activeKnot(ii) = 1;
    end
end
% Caculate Ni,j for higher degree 1 ,2 ,3.....
for ii = 1:Intervals  %each interval
    if(activeKnot(ii)~=0)
       for jj=1:Degree %each degree or Ni,j
          id0 = Order*sum(0:(jj-1))+1;
          id1 = Order*sum(0:(jj))+1;
          for kk = 0:jj %each id member matrix at iterval ii
             id2 = ii-jj+kk; %effective internal we focus,eg when ii=4,jj=1,then id 2=3,4
             id2Knot00 = id2 + jj; % effective knot num 1
             id2Knot01 = id2Knot00 + 1;
             if (id2>0)&&(id2Knot01<=numel(Knot));
                 % Access previous data Ni-1,j-1 Ni,j-1 and Ni+1,j-1
                 id00 = id0 + (kk-1)*Order;
                 id01 = id0+kk*Order;
                 if kk==0 %first box of matrix
                     N0 = zeros(Order,1);
                     N1 = CoeNmat(id01:(id01+Degree),ii);
                 elseif kk==(jj)
                     N0 = CoeNmat(id00:id00+Degree,ii);
                     N1 = zeros(Order,1);
                 else
                     N0= CoeNmat(id00:id00+Degree,ii);
                     N1= CoeNmat(id01:(id01+Degree),ii);                    
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
                 CoeNmat(id11:id11+Degree,ii)=Acoef;
              end
          end      
      end
    end   
end
 % Store data
id10 = Order*sum(0:(Degree))+1;
Nmat = CoeNmat(id10:end,:);
end

