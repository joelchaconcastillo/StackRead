%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name: TwoPieceOptimalKnotSolver.m
% Description: This function finds optimal knot and its continuity. The
% input data must fully define each single pieces
% Prototype: 
% [OptimalKnotOut,MultipleOut]= TwoPieceOptimalKnotSolver(DataIn,Degree,...
%       SearchRange,MinAngle,N0ScanForDiscontinuosCase,GaussNewtonLoopTime)
% Input parameters:
% - DataIn: A matrix contains input data, having at least 2 columns. 1st
% column is parametric, 2nd column is X, 3rd column is Y and so on.
% - Degree: Degree of the fitted B-spline
% - SearchRange: [a,b] is a range to find optimal knot
% - MinAngle: Minimum joining angle at knot that we can accept the finding
% optimal knot.
% - N0ScanForDiscontinuosCase: Number of evaluation times in calculating
% discontinuity case.
% - GaussNewtonLoopTime: Maximum number of loops in Gauss-Newton solving.
% Output parameters:
% - OptimalKnotOut: Optimum knot location
% - MultipleOut: Multiple knot at the optimal knot join. If there is no
% optimal knot found, this variable will return 0 (Eliminate the join)
% error.
% Version: 1.1                                                             
% Date: 19-July-2016
% Author: Dvthan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [OptimalKnotOut,MultipleOut] = TwoPieceOptimalKnotSolver1(DataIn,DataKnot,...
    Degree,MinAngle,MaxSmooth,N0ScanForDiscontinuosCase,GaussNewtonLoopTime,ScanKnot)

Order = Degree + 1;
[m,n] = size(DataIn);

T1 = DataIn(:,1);
T = zeros(numel(T1),Order);
T(:,1) = 1;
for cnt = 2:Order
    T(:,cnt) = T(:,cnt-1).*T1;
end
Ymat = DataIn(:,2:end);

idx01 = DataKnot;
idx10 = DataKnot + 1;
idx00 = 1;
idx11 = m;
DP1 = idx01 - idx00 - Degree;
DP2 = idx11 - idx10 - Degree;
LeftSearch = zeros(1,Order);
RightSearch = zeros(1,Order);
for cnt = 1:Order
    LeftSearch(1,cnt) = min(DP1,3)+Degree-cnt;
    RightSearch(1,cnt) = min(DP2,3)+Degree-cnt;
end
% if (DP1>0)&&(DP2>0)
%     LeftSearch(1,Order) = LeftSearch(1,Order) + 1;
%     RightSearch(1,Order) = RightSearch(1,Order) + 1;
% elseif (DP1<=0)&&(DP2>0)
%     RightSearch(1,Order) = RightSearch(1,Order) + 1;
% elseif (DP2<=0)&&(DP1>0)
%     LeftSearch(1,Order) = LeftSearch(1,Order) + 1;
% end
%% Uniform scanning the lowest error
OptimalKnotSave = zeros(1,Order);
ErrorSave = zeros(1,Order);
AngleSave = zeros(1,Order);
SearchRangeOut = zeros(2,Order);
MultipleMax = Order;
for ii = Order:-1:1
%     LeftSearch1 = max(LeftSearch(ii),0);
%     RightSearch1 = max(RightSearch(ii),0);
    LeftSearch1 = LeftSearch(ii);
    RightSearch1 = RightSearch(ii);
    SearchRange = [DataIn(max(idx01-LeftSearch1,1),1),DataIn(min(idx10+RightSearch1,idx11),1)];
    if (SearchRange(2)>SearchRange(1))
        ExpandRange = max(ceil(0.5*(N0ScanForDiscontinuosCase+1)/(LeftSearch1+RightSearch1)),1);    
        Multiple = ii;
        KnotLocation = SearchRange(1):((SearchRange(2)-SearchRange(1))/N0ScanForDiscontinuosCase):SearchRange(2);
        DisErrorSave = zeros(1,numel(KnotLocation));
        DisAnglePsave = zeros(1,numel(KnotLocation));
        for cnt = 1:numel(KnotLocation)
            StartPoint = KnotLocation(cnt);
            [~,DisErrorSave(cnt),DisAnglePsave(cnt)] = TwoPieceBspineKnotEval1(T,Ymat,Degree,Multiple,StartPoint);
        end
        % compute fourfold knot position
        KnotLocation = KnotLocation(~isnan(DisAnglePsave));
        DisErrorSave = DisErrorSave(~isnan(DisAnglePsave));
        DisAnglePsave = DisAnglePsave(~isnan(DisAnglePsave));
        %% Select region 
        if ii == Order
            CtrlError1 = min(DisErrorSave)+1e-10;
            disIdx = find(DisErrorSave<CtrlError1);
            MidPosition = round(0.5*(min(disIdx)+max(disIdx)));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            SearchRangeOut(:,ii) = [KnotLocation(max(disIdx(1)-1,1));KnotLocation(min(disIdx(end)+1,numel(KnotLocation)))];
            OptimalKnotSave(ii) = KnotLocation(MidPosition);
            AngleSave(ii) = DisAnglePsave(MidPosition);
            ErrorSave(ii) = DisErrorSave(MidPosition);
            if ~ScanKnot
                OptimalKnotSave(1:Degree) = OptimalKnotSave(ii);
                for cnt = 1:Degree
                    SearchRangeOut(:,cnt) = [KnotLocation(max(disIdx(1)-1,1));KnotLocation(min(disIdx(end)+1,numel(KnotLocation)))];
                end
                break;        
            end
        else
            [MinError,disIdx] = min(DisErrorSave);
            SearchRangeOut(:,ii) = [KnotLocation(max(disIdx-ExpandRange,1));KnotLocation(min(disIdx+ExpandRange,numel(KnotLocation)))];
            OptimalKnotSave(ii) = KnotLocation(disIdx);
            AngleSave(ii) = DisAnglePsave(disIdx);
            ErrorSave(ii) = MinError;
        end        
    else
        MultipleMax = MultipleMax - 1;
        ScanKnot = 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cnt = 1:(min(MultipleMax,Degree))
    StartPoint = OptimalKnotSave(cnt);
    SearchRange = SearchRangeOut(:,cnt)';
    Multiple = cnt;
    [OptimalKnotO,ErrorO,~,~,Angle] = GNKnotSolver1(T,Ymat,...
    Degree,Multiple,SearchRange,StartPoint, GaussNewtonLoopTime);  
    if (ScanKnot)&&(MultipleMax==Order)
        StartPoint = OptimalKnotSave(Order);        
        SearchRange = SearchRangeOut(:,Order)';
        [OptimalKnotOs,ErrorOs,~,~,Angles] = GNKnotSolver1(T,Ymat,...
        Degree,Multiple,SearchRange,StartPoint, GaussNewtonLoopTime);
        if ErrorOs<ErrorO
            OptimalKnotO = OptimalKnotOs;
            ErrorO = ErrorOs;
            Angle = Angles;
        end        
    end
    OptimalKnotSave(cnt) = OptimalKnotO;
    ErrorSave(cnt) = ErrorO;
    AngleSave(cnt) = Angle;
end
%% decide multiple knot
SmoothOutput = Degree - MaxSmooth;
SmoothOutput = min(SmoothOutput,MultipleMax);
AngleOut = AngleSave(1:SmoothOutput);
idx1 = find(AngleOut>MinAngle);

if numel(idx1)
    ErrorOut = ErrorSave(idx1);
    [~, minidx] = min(ErrorOut);
    OptimalIdx = idx1(minidx);
    
    OptimalKnotOut = OptimalKnotSave(OptimalIdx);
    MultipleOut = OptimalIdx; 
else
    % eliminate the knot
    [~,idx] = max(AngleOut);
    OptimalKnotOut = OptimalKnotSave(idx);
    MultipleOut = idx; 
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subfunctions
%% Find error of a knot
function [OptimalKnot,Error,JoinAngle] = TwoPieceBspineKnotEval1(T,Ymat,Degree,Multiple,StartPoint)
    OptimalKnot = StartPoint;
    Order = Degree + 1;
    T1 = T(:,2);
    [datalength,datasize12] = size(Ymat);
%     datasize12 = datasize12 + 1;
    % Derivative
    Tcoef = ones(Order,Order);
    for ii = Degree:-1:1
        for cnt = 1:Order
            mul = (cnt - (Order -ii));
            if mul<0
                mul = 0;
            end
            Tcoef(ii,cnt) = Tcoef(ii+1,cnt)*mul;
        end
    end
    Tcal = ones(1,Order);
    for cnt = 2:Order
        Tcal(cnt) = Tcal(cnt-1)*OptimalKnot;
    end    
    Tcal = Tcal.*Tcoef(Multiple,:);
    % B-spline evaluation
    Knot = zeros(1,2*Order+Multiple);
    Knot(1:Order) = T1(1);
    Knot((Order+Multiple + 1):end) = T1(end);
    Knot((Order+1):(Order+Multiple))= OptimalKnot;
    left = 1;
    right = datalength;                             
    middle = fix(0.5*(left+right)); 
    loopknot = ceil(log2(datalength));
    for cnt = 1:loopknot
        if OptimalKnot < T1(middle)
            right = middle;
        else
            left = middle;
        end                              
        middle = fix(0.5*(left+right));                                     
    end                
    % update Nmat
    Nmat = NewNmatrix(Knot,Degree);  
    % generate the knot1 for G'mat
    % Compute Basis function base an KnotL and Knot1L
    
    N = zeros(datalength,Order+Multiple);                
    Ncal = reshape(Nmat(:,Order),[Order,Order]);
    Sleft = Tcal*Ncal;
    N(1:middle,1:Order) = T(1:middle,:)*Ncal;
    Ncal = reshape(Nmat(:,Order + Multiple),[Order,Order]);
    Sright = Tcal*Ncal;
    N(middle+1:end,1+Multiple:(Multiple+Order)) = T(middle+1:end,:)*Ncal;  
    if datasize12 == 1
        Pctrl = N(:,1:(Multiple+Order))\Ymat;               
        Gmat = abs(Ymat-N(:,1:(Multiple+Order))*Pctrl);                
    else
        Pctrl = N(:,1:(Multiple+Order))\Ymat;               
        G = (Ymat-N(:,1:(Multiple+Order))*Pctrl);                
        G = G.*G;               
        Gmat = sum(G,2);
        Gmat = sqrt(Gmat);                
    end    
    Error = max(abs(Gmat));    
    %% Calculating join angle    
    Sleft1 = Sleft*Pctrl(1:Order,:);
    Sright1 = Sright*Pctrl(Multiple + 1:Multiple+Order,:);
    if datasize12 == 1
        JoinAngle = abs(atan(Sright1/OptimalKnot)-atan(Sleft1/OptimalKnot));
    else
        CosPhi = dot(Sleft1,Sright1)/(norm(Sleft1)*norm(Sright1));
        JoinAngle = acos(CosPhi);        
    end
    JoinAngle = JoinAngle*180/pi();
end
%% Gauss Newton solver
function [OptimalKnotO,ErrorO,OptimalKnot,Error,Angle] = GNKnotSolver1(T,Ymat,...
    Degree,Multiple,SearchRange,StartPoint, GaussNewtonLoopTime)   
    T1 = T(:,2);    
    Stepsize = sqrt(eps);
    Order = Degree + 1;    
    [datalength,datasize12] = size(Ymat);
%     datasize12 = datasize12 + 1;
    KnotLeft = SearchRange(1);
    KnotRight = SearchRange(2);
    loopknot = ceil(log2(datalength));
    
    OptimalKnot = StartPoint;
    Knot = zeros(1, 2*Order +  Multiple);
    Knot(1:Order) = T1(1);
    Knot(Order+Multiple+1:end) = T1(end);
    Knot1 = Knot;
    % Derivative
    Tcoef = ones(Order,Order);
    for ii = Degree:-1:1
        for cnt = 1:Order
            mul = (cnt - (Order -ii));
            if mul<0
                mul = 0;
            end
            Tcoef(ii,cnt) = Tcoef(ii+1,cnt)*mul;
        end
    end
    
    
    for iterationstep = 1:GaussNewtonLoopTime
        % generate the knot based on the multiple type                 
        Knot(Order+1:Order+Multiple)= OptimalKnot;
        OptimalKnot1 = OptimalKnot + Stepsize;
        Knot1(Order+1:Order+Multiple)= OptimalKnot1;                   
        % find knot location
        left = 1;
        right = datalength;        
        left1 = left;
        right1 = right;                  
        middle = fix(0.5*(left+right));
        middle1 = fix(0.5*(left1+right1));  
        
        for cnt = 1:loopknot
            if OptimalKnot < T1(middle)
                right = middle;
            else
                left = middle;
            end
            if OptimalKnot1 < T1(middle1)
                right1 = middle1;
            else
                left1 = middle1;
            end                    
            middle = fix(0.5*(left+right));
            middle1 = fix(0.5*(left1+right1));                    
        end                
        % update Nmat
        Nmat = NewNmatrix(Knot,Degree);                
        Nmat1 = NewNmatrix(Knot1,Degree);
        % generate the knot1 for G'mat
        Tcal = ones(1,Order);
        for cnt = 2:Order
            Tcal(cnt) = Tcal(cnt-1)*OptimalKnot;
        end    
        Tcal = Tcal.*Tcoef(Multiple,:);
        % Compute Basis function base an KnotL and Knot1L
        N = zeros(datalength,Order+Multiple);
        %N(datalength,Order+Multiple) = 0;
        N1 = N;
        Ncal = reshape(Nmat(:,Order),[Order,Order]);
        Sleft = Tcal*Ncal;
        N(1:middle,1:Order) = T(1:middle,:)*Ncal;
        Ncal = reshape(Nmat(:,Order + Multiple),[Order,Order]);
        N(middle+1:end,1+Multiple:(Multiple+Order)) = T(middle+1:end,:)*Ncal;  
        Sright = Tcal*Ncal;
        
        Ncal = reshape(Nmat1(:,Order),[Order,Order]);
        N1(1:middle1,1:Order) = T(1:middle1,:)*Ncal;
        Ncal = reshape(Nmat1(:,Order + Multiple),[Order,Order]);
        N1(middle1+1:end,1+Multiple:(Multiple+Order)) = T(middle1+1:end,:)*Ncal;
        
        if datasize12 == 1
            Pctrl = N\Ymat;
            Pctrl1 = N1\Ymat;
            Gmat = abs(Ymat-N*Pctrl);
            Gmat1 = abs(Ymat-N1*Pctrl1);
        else
            Pctrl = N\Ymat;
            Pctrl1 = N1\Ymat;
            G = (Ymat-N*Pctrl);
            G1 = (Ymat-N1*Pctrl1);
            G = G.*G;
            G1 = G1.*G1;
            Gmat = sum(G,2);
            Gmat = sqrt(Gmat);
            Gmat1 = sum(G1,2);
            Gmat1 = sqrt(Gmat1);
        end                
        % Compute Jacobian matrix G'mat
        Jmat = (Gmat1-Gmat)/Stepsize; % size Xlength x 1
        deltaX = (Jmat'*Jmat)^-1*Jmat'*Gmat;
        if iterationstep == 1
            deltaXLOld = deltaX;
        else
            if (deltaXLOld*deltaX < 0)
                deltaX = 0.5*deltaX;
            end
            deltaXLOld = deltaX;
        end
        OptimalKnot = OptimalKnot - deltaX;
        % calculating angle
        Sleft1 = Sleft*Pctrl(1:Order,:);
        Sright1 = Sright*Pctrl(Multiple + 1:Multiple+Order,:);
        if datasize12 == 1
            JoinAngle = abs(atan(Sright1/OptimalKnot)-atan(Sleft1/OptimalKnot));
        else
            CosPhi = dot(Sleft1,Sright1)/(norm(Sleft1)*norm(Sright1));
            JoinAngle = acos(CosPhi);        
        end
        JoinAngle = JoinAngle*180/pi();
        % saturation Optimal knot
        if OptimalKnot < KnotLeft
            OptimalKnot = KnotLeft;
        end
        if OptimalKnot > KnotRight
            OptimalKnot = KnotRight;
        end                  
        Error = max(abs(Gmat));  
        % check for stop
        if iterationstep > 1
            if Error < ErrorO
                OptimalKnotO = OptimalKnot; 
                ErrorO = Error;
                Angle = JoinAngle;            
            end
            if (abs(OptimalKnotLOld-OptimalKnot)< 1e-12)
                if checkflag==0
                    checkflag = 1;
                else                                 
                    break;
                end                    
            else
                OptimalKnotLOld = OptimalKnot;                        
                checkflag = 0;
            end
        else
            OptimalKnotLOld = OptimalKnot;                     
            OptimalKnotO = OptimalKnot; 
            ErrorO = Error;
            Angle = JoinAngle;
            checkflag = 0;
        end    
    end
    % calculating join angle
    
end
%% NewNmatrix
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