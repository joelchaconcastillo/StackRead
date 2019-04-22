%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file name: ParallelBisection.m
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
% - NumOfPieceStart: Number of b-spline pieces when starting.
% Output parameters:
% - VectorUX: a matrix contains information of each single piece b-spline.
% each column stores the information of each single piece b-spline.
% 1st row: start index, 2nd: end index, 3 to 2+Order^2: Coefficient vector,
% 3 + Order^2 to end - 1: B-spline control points, end: maximum fitted
% error.
% Version: 1.0                                                             
% Date: 1-July-2016
% Author: Dvthan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function VectorUX = ParallelBisection1(DataIn,Degree,CtrlError,NumOfPieceStart)
%% Global variable
[N,S] = size(DataIn);
if nargin < 4
    NumOfPieceStart = max(round (N/(8*Degree)),2);    
end
Order = Degree + 1;
Order2= Order*Order;
Amat = zeros(N,Order);
Amat(:,1) = 1;
for ii = 2:Order
    Amat(:,ii) = Amat(:,ii-1).*DataIn(:,1);
end
Ymat = DataIn(:,2:end);
ErrorBandLimit = CtrlError*CtrlError;
%% half split method 
StartT = 1;
EndT = N+1;
KnotX1 = StartT:(round(2*(EndT-1)/NumOfPieceStart)):EndT;  
KnotX1(end) = EndT;
DeltaX = max(diff(KnotX1));
maxdistance2X0 = ErrorBandLimit + 1000;
NvectorUX = 3+Order*(S-1)+Order2;
VectorUX = zeros(NvectorUX,length(KnotX1)-1);
VectorUX(end,:)= maxdistance2X0;
DivideNeedX = ones(1,length(KnotX1)-1);

%% Half split procedure
NumberofRepeat = 1;
while (1)
    piecenumX = numel(KnotX1)-1;
    %% half split KnotX1
    DeltaX = ceil(DeltaX/2);
    Ttemplength = 2*piecenumX+1;
    KnotX1temp = zeros(1,Ttemplength);
    PreviousSegmenttemp = zeros(1,Ttemplength);
    DivideNeedXoldtemp = zeros(1,Ttemplength);
    indexknot = 1;
    for ii = 1:piecenumX
        piecespace = KnotX1(ii+1)- KnotX1(ii);
        if piecespace >= (2*Order) % half split
            if DivideNeedX(ii)== 1
                KnotX1temp(indexknot)= KnotX1(ii);
                PreviousSegmenttemp(indexknot)= ii;
                DivideNeedXoldtemp(indexknot) = 1;
                indexknot = indexknot + 1;
                KnotX1temp(indexknot)= round(0.5*(KnotX1(ii) + KnotX1(ii+1)));
                PreviousSegmenttemp(indexknot)= ii;
                DivideNeedXoldtemp(indexknot) = 1;
                indexknot = indexknot + 1;
            else
                KnotX1temp(indexknot)= KnotX1(ii);
                PreviousSegmenttemp(indexknot)= ii;
                DivideNeedXoldtemp(indexknot) = 0;
                indexknot = indexknot + 1;
            end
        else % do not half split, keep the piece
            KnotX1temp(indexknot)= KnotX1(ii);
            PreviousSegmenttemp(indexknot)= ii; 
            DivideNeedXoldtemp(indexknot) = DivideNeedX(ii);
            indexknot = indexknot + 1;
        end
    end
    KnotX1temp(indexknot)= KnotX1(end);
    DivideNeedXold = DivideNeedXoldtemp(1:indexknot-1);
    PreviousSegment = PreviousSegmenttemp(1:indexknot-1);
    KnotX1 = KnotX1temp(1:indexknot);
    %% Compute new error band    
    piecenumX = length(KnotX1)-1;
    VectorUXold = VectorUX;
    VectorUX = zeros(NvectorUX,piecenumX); 
    VectorUX(:,~DivideNeedXold) = VectorUXold(:,PreviousSegment(~DivideNeedXold)); 
    % Par for can be used here    
    for ii = 1: piecenumX % process for all piecewise
        if (DivideNeedXold(ii)~=0)
            index0 = KnotX1(ii);
            index1 = KnotX1(ii+1)-1;
            if (index1-index0)>= Degree
                Y = Ymat(index0:index1,:);
                A = Amat(index0:index1,:);
                % Single piece Nmatrix cal
                Knotin = [DataIn(index0,1),DataIn(index1,1)];
                SPNmat = NmatCal_1P(Knotin,Degree);
                NT = A*reshape(SPNmat,[Order,Order]);
                CtrlP = NT\Y;
                
                VectorUX(1,ii) = index0; % startptr
                VectorUX(2,ii) = index1; % stopptr  
                VectorUX(3:2+Order2,ii) = SPNmat; % Nmatrix
                VectorUX(3+Order2:end-1,ii) = reshape(CtrlP,[Order*(S-1),1]);

                % Error band
                % Error band calculation                
                R = Y-NT*CtrlP;
                R = R.*R;
                D2max = max(sum(R,2));
                
                VectorUX(end,ii) = D2max;    
            else %in case the datalength does not enough for finding approxation function
                VectorUX(1,ii) = index0; % startptr
                VectorUX(2,ii) = index1; % stopptr
                VectorUX(end,ii) = 0;
            end          
        end
    end
    %% Compare error band and create divideneedX
    DivideNeedX = zeros(1,piecenumX);
    distance2 = VectorUX(end,:);  
    DivideNeedX(distance2>ErrorBandLimit) = 1;
    %% Joining knot and creating new knot vector 
    VectorUXJoin = zeros(NvectorUX,piecenumX);
    if NumberofRepeat == 1
        JoinLocation = find(~DivideNeedX); % find change sign location
        PreviousSegment = 1:numel(PreviousSegment);
    else
        JoinLocation = find(xor(DivideNeedX,DivideNeedXold)); % find change sign location
    end    
    if numel(JoinLocation)>=1
        ii = 1;
        jj = 1;
        for kk = 1:numel(JoinLocation) % check for join with left side and right side
            idx0 = JoinLocation(kk);
            % join with the left side
            if (idx0 > 1)
                leftflag = 1;
                if (kk > 1)
                    if (JoinLocation(kk)-JoinLocation(kk-1))==1
                        leftflag = 0;
                    end
                end   
                if (leftflag ~= 0)
                    if (~DivideNeedX(idx0-1))&&(PreviousSegment(idx0)~=PreviousSegment(idx0-1))
                        idx2 = idx0;
                        % write VectorUXjoin
                        if idx0>jj
                            datafilsize = idx0-jj;                    
                            VectorUXJoin(:,ii:ii+datafilsize-1) = VectorUX(:,jj:idx0-1);
                            ii = ii+datafilsize;
                            jj = idx0;
                        end
                        % Computing new join
                        idx10 = VectorUXJoin(1,ii-1);
                        idx20 = VectorUX(2,idx2);

                        Y = Ymat(idx10:idx20,:);
                        A = Amat(idx10:idx20,:);
                        % Single piece Nmatrix cal
                        Knotin = [DataIn(idx10,1),DataIn(idx20,1)];
                        SPNmat = NmatCal_1P(Knotin,Degree);
                        NT = A*reshape(SPNmat,[Order,Order]);
                        CtrlP = NT\Y;     
                        % Error band
                        % Error band calculation                        
                        R = Y-NT*CtrlP;
                        R = R.*R;
                        D2max = max(sum(R,2));
                        
                        if D2max > ErrorBandLimit
                            % cannot join keep original pieces                    
                            VectorUXJoin(:,ii) = VectorUX(:,idx0);
                            ii = ii+1;
                            jj = idx2+1;
                        else
                            % join                    
                            VectorUXJoin(1,ii-1) = idx10; % startptr
                            VectorUXJoin(2,ii-1) = idx20; % stopptr  
                            VectorUXJoin(3:2+Order2,ii-1) = SPNmat; % Nmatrix
                            VectorUXJoin(3+Order2:end-1,ii-1) = reshape(CtrlP,[Order*(S-1),1]);
                            VectorUXJoin(end,ii-1) = D2max; 
                            ii = ii;
                            jj = idx2+1;
                        end
                    end
                end
            end   
            % join with the right side
            if idx0 < numel(DivideNeedX)
                if (~DivideNeedX(idx0+1))&&(PreviousSegment(idx0)~=PreviousSegment(idx0+1))
                    idx2 = idx0+1;
                    % write VectorUXjoin
                    if idx2>jj
                        datafilsize = idx2-jj;                    
                        VectorUXJoin(:,ii:ii+datafilsize-1) = VectorUX(:,jj:idx2-1);
                        ii = ii+datafilsize;
                        jj = idx2;
                    end
                    % Computing new join
                    idx10 = VectorUXJoin(1,ii-1);
                    idx20 = VectorUX(2,idx2);

                    Y = Ymat(idx10:idx20,:);
                    A = Amat(idx10:idx20,:);
                    % Single piece Nmatrix cal
                    Knotin = [DataIn(idx10,1),DataIn(idx20,1)];
                    SPNmat = NmatCal_1P(Knotin,Degree);
                    NT = A*reshape(SPNmat,[Order,Order]);
                    CtrlP = NT\Y;     
                    % Error band
                    % Error band calculation                    
                    R = Y-NT*CtrlP;
                    R = R.*R;
                    D2max = max(sum(R,2)); 
                    
                    if D2max > ErrorBandLimit
                        % cannot join keep original pieces                    
                        VectorUXJoin(:,ii) = VectorUX(:,idx0+1);
                        ii = ii+1;
                        jj = idx2+1;
                    else
                        % join                    
                        VectorUXJoin(1,ii-1) = idx10; % startptr
                        VectorUXJoin(2,ii-1) = idx20; % stopptr  
                        VectorUXJoin(3:2+Order2,ii-1) = SPNmat; % Nmatrix
                        VectorUXJoin(3+Order2:end-1,ii-1) = reshape(CtrlP,[Order*(S-1),1]);
                        VectorUXJoin(end,ii-1) = D2max; 
                        ii = ii;
                        jj = idx2+1;
                    end
                end
            end   
            % fill the remaining
            if kk == numel(JoinLocation)
                if jj<=piecenumX
                    datafilsize = piecenumX-jj;                    
                    VectorUXJoin(:,ii:ii+datafilsize) = VectorUX(:,jj:piecenumX);
                    ii = ii + datafilsize + 1;
                end
            end
            
        end
        %clear KnotX1 VectorUX DivideNeedX
        KnotX1 = zeros(1,ii);
        ii = ii - 1;        
        KnotX1(1:ii) = VectorUXJoin(1,1:ii);
        KnotX1(end) = VectorUXJoin(2,ii)+1;        
        VectorUX = VectorUXJoin(:,1:ii);
        DivideNeedX = zeros(1, ii);
        newerror =  VectorUX(end,:);
        DivideNeedX(newerror>ErrorBandLimit) = 1; 
    end
    %% Shift the small piece
    if DeltaX < (2*Order)
        Dknot = diff(KnotX1);
        SmallPieces = find(Dknot<(2*Order)); 
        looptimes = numel(SmallPieces);
        if looptimes>0
            [vn,vm] = size(VectorUX);
            VectorUXJoin = zeros(vn,vm);
            ii = 1;
            jj = 1;
            for k = 1:looptimes
                % Copy large pieces
                idx0 = SmallPieces(k);
                if idx0>jj
                    datafilsize = idx0-jj;                    
                    VectorUXJoin(:,ii:ii+datafilsize-1) = VectorUX(:,jj:idx0-1);
                    ii = ii+datafilsize;
                    jj = idx0;
                end 
                %% shifting small pieces                
                if  ii > 1  % replace idx0 by ii
                    %% shift the left side to the right 
                    % shift the left side piece to the right to eliminate x 
                    index00 = VectorUXJoin(1,ii-1);
                    index01 = VectorUXJoin(2,ii-1);
                    index2 = VectorUX(2,idx0);
                    if (index2-index00)>Degree
                        if (index01-index00)>=Degree
                            idxleft = index01;
                        else
                            idxleft = index00+Degree;
                        end                        
                        idxright = index2;                         
                        while(idxright>(idxleft+1))    
                            index1 = ceil(0.5*(idxright+idxleft));
                            Y = Ymat(index00:index1,:);
                            A = Amat(index00:index1,:); 
                            Knotin = [DataIn(index00,1),DataIn(index1,1)];
                            SPNmat = NmatCal_1P(Knotin,Degree);
                            % Single piece Nmatrix cal
                            NT = A*reshape(SPNmat,[Order,Order]);
                            CtrlP = NT\Y;
                            % compute new error
                            R = Y-NT*CtrlP;
                            R = R.*R;
                            D2max = max(sum(R,2)); 
                            if (D2max > ErrorBandLimit)                                
                                idxright = index1;                                 
                            else
                                idxleft = index1;
                                VectorUXJoin(3:2+Order2,ii-1) = SPNmat; % Nmatrix
                                VectorUXJoin(3+Order2:end-1,ii-1) = reshape(CtrlP,[Order*(S-1),1]);
                                VectorUXJoin(end,ii-1) = D2max; 
                            end    
                        end    
                        if (D2max > ErrorBandLimit)
                            % modify the left piece
                            VectorUXJoin(2,ii-1)=index1-1;                    
                            % modify the current piece
                            VectorUX(1,idx0)=index1; 
                        else
                            % modify the left piece
                            VectorUXJoin(2,ii-1)=index1;                    
                            % modify the current piece
                            VectorUX(1,idx0)=index1 +1 ;
                        end
                    else
                        % modify the left piece
                        VectorUXJoin(2,ii-1)=index2;                    
                        % modify the current piece
                        VectorUX(1,idx0)= index2 + 1; 
                    end
                end 
                if idx0 < numel(Dknot)
                    %% shift the right side to the left
                    VectorUXJoin(:,ii) = VectorUX(:,jj);
                    ii = ii + 1;
                    jj = jj + 1;
                    index00 = VectorUXJoin(1,ii-1);
                    index01 = VectorUX(1,idx0 + 1);
                    index2 = VectorUX(2,idx0 + 1);
                    if (index2-index00)>Degree
                        idxleft = index00;
                        idxright = index01;                         
                        while(idxright>(idxleft+1))    
                            index1 = fix(0.5*(idxright+idxleft));
                            Y = Ymat(index1:index2,:);
                            A = Amat(index1:index2,:); 
                            Knotin = [DataIn(index1,1),DataIn(index2,1)];
                            SPNmat = NmatCal_1P(Knotin,Degree);
                            % Single piece Nmatrix cal
                            NT = A*reshape(SPNmat,[Order,Order]);
                            CtrlP = NT\Y;
                            % compute new error
                            R = Y-NT*CtrlP;
                            R = R.*R;
                            D2max = max(sum(R,2)); 
                            if (D2max > ErrorBandLimit) 
                                idxleft = index1;
                            else
                                idxright = index1; 
                                VectorUX(3:2+Order2,idx0+1) = SPNmat; % Nmatrix
                                VectorUX(3+Order2:end-1,idx0+1) = reshape(CtrlP,[Order*(S-1),1]);
                                VectorUX(end,idx0+1) = D2max; 
                            end    
                        end    
                        if (D2max > ErrorBandLimit)
                            % modify the left piece
                            VectorUXJoin(2,ii-1)=index1;                    
                            % modify the current piece
                            VectorUX(1,idx0+1)=index1+1; 
                            if index1<=index00
                                ii = ii - 1;
                                VectorUX(1,idx0+1)=index00;  
                            end
                        else
                            % modify the left piece
                            VectorUXJoin(2,ii-1)=index1-1;                    
                            % modify the current piece
                            VectorUX(1,idx0+1)=index1;                            
                            if (index1-1)<=index00
                                ii = ii - 1;
                                VectorUX(1,idx0+1)=index00;  
                            end
                        end
                    else
                        % modify the left piece
                        ii = ii - 1;                   
                        % modify the current piece
                        VectorUX(1,idx0+1)= index00; 
                    end
                 end    
                if k == looptimes
                    idx0 = numel(Dknot);
                    if idx0>=jj
                        datafilsize = idx0-jj;                    
                        VectorUXJoin(:,ii:ii+datafilsize) = VectorUX(:,jj:idx0);
                        ii = ii+datafilsize+1;
                        jj = idx0+1;
                    end
                end
            end
            ii = ii -1;
            VectorUX = VectorUXJoin(:,1:ii);
            break;
        end
    end
    %% break if all the piece satisfy the condition
    if DeltaX <= 1
        break;
    end
    NumberofRepeat = NumberofRepeat + 1;
end
% trying to expand small pieces
[~,m] = size(VectorUX);
for ii = 1:m
    index0 = VectorUX(1,ii);
    index1 = VectorUX(2,ii);
    if (index1-index0)<(2*Order-1)
        if ii > 1
            index0 = index0 - 1;
            if (index1-index0)>= Degree
                Y = Ymat(index0:index1,:);
                A = Amat(index0:index1,:);
                % Single piece Nmatrix cal
                Knotin = [DataIn(index0,1),DataIn(index1,1)];
                SPNmat = NmatCal_1P(Knotin,Degree);
                NT = A*reshape(SPNmat,[Order,Order]);
                CtrlP = NT\Y;     
                % Error band
                % Error band calculation                    
                R = Y-NT*CtrlP;
                R = R.*R;
                D2max = max(sum(R,2));
                if D2max <= ErrorBandLimit 
                    VectorUX(2,ii-1) = index0-1; % startptr
                    VectorUX(1,ii) = index0; % startptr
                    VectorUX(2,ii) = index1; % stopptr  
                    VectorUX(3:2+Order2,ii) = SPNmat; % Nmatrix
                    VectorUX(3+Order2:end-1,ii) = reshape(CtrlP,[Order*(S-1),1]);
                    VectorUX(end,ii) = D2max;                     
                else
                    index0 = index0 + 1;
                end
            end
        end
        if ii < m
            index1 = index1 + 1;
            if (index1-index0)>= Degree
                Y = Ymat(index0:index1,:);
                A = Amat(index0:index1,:);
                % Single piece Nmatrix cal
                Knotin = [DataIn(index0,1),DataIn(index1,1)];
                SPNmat = NmatCal_1P(Knotin,Degree);
                NT = A*reshape(SPNmat,[Order,Order]);
                CtrlP = NT\Y;     
                % Error band
                % Error band calculation                    
                R = Y-NT*CtrlP;
                R = R.*R;
                D2max = max(sum(R,2));
                if D2max <= ErrorBandLimit 
                    VectorUX(1,ii+1) = index1+1; % startptr
                    VectorUX(1,ii) = index0; % startptr
                    VectorUX(2,ii) = index1; % stopptr  
                    VectorUX(3:2+Order2,ii) = SPNmat; % Nmatrix
                    VectorUX(3+Order2:end-1,ii) = reshape(CtrlP,[Order*(S-1),1]);
                    VectorUX(end,ii) = D2max;                     
                else
                    index0 = index0 + 1;
                end
            end
        end
    end    
end
end
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