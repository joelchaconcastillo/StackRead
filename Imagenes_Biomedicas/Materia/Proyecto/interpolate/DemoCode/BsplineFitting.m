function [BSpline,Error,FittedData] = BsplineFitting(OptimalKnotOut,MultipleOut,DataIn,Degree)

Order = Degree + 1;
MultipleKnot = MultipleOut(MultipleOut>0);
[N,~] = size(DataIn);
Amat = zeros(N,Order);
Amat(:,1) = 1;
for ii = 2:Order
    Amat(:,ii) = Amat(:,ii-1).*DataIn(:,1);
end
Ymat = DataIn(:,2:end);

Knot = zeros(1,2*Order + sum(MultipleKnot));
Knot(1:Order) = DataIn(1,1);
Knot(end-Degree:end) = DataIn(end,1);
jj = Order + 1;
for ii = 1:numel(OptimalKnotOut)
    for kk = 1:MultipleOut(ii)
        Knot(jj) = OptimalKnotOut(ii);
        jj = jj + 1;
    end
end
%% B-spline fitting
% find knot index
Knotout = OptimalKnotOut(MultipleOut>0);

looptime = numel(Knotout);
Numloop = ceil(log2(N));
InteriorKnotIdx = zeros(looptime,1);
for k=1:looptime
    stidx = 1;
    endidx = N;
    mididx = fix(0.5*(stidx+endidx));
    for cnt = 1:Numloop
        if Knotout(k)<=DataIn(mididx,1)
            endidx = mididx;
        else
            stidx = mididx;
        end
        mididx = fix(0.5*(stidx+endidx));
    end
    InteriorKnotIdx(k)=mididx;
end
Nmat = NewNmatrix(Knot,Degree);
ANmat = zeros(N,numel(Knot)-Order);
Dknot = diff(Knot);
indexnonzero=find(Dknot);
idx0 = 1;
idxrow0 = 1;
for k=1:(looptime)
    idx1 = InteriorKnotIdx(k);
    idxrow1 = idxrow0 + Degree;
    ANmat(idx0:idx1,idxrow0:idxrow1)=Amat(idx0:idx1,:)...
        *reshape(Nmat(:,indexnonzero(k)),[Order,Order]);    
    idx0 = idx1 + 1;
    idxrow0 = idxrow0 + MultipleKnot(k);
end
idx1 = N;
idxrow1 = idxrow0 + Degree;
ANmat(idx0:idx1,idxrow0:idxrow1)=Amat(idx0:idx1,:)...
        *reshape(Nmat(:,indexnonzero(k+1)),[Order,Order]);   
    
CtrlPoints = ANmat\Ymat;
Yfit = ANmat*CtrlPoints;
R = Ymat - Yfit;
Error = sqrt(sum(R.*R,2));
% MaxError = max(Error);
% return B-spline
BSpline.knot = Knot;
BSpline.degree = Degree;
BSpline.ctrlp = CtrlPoints;
BSpline.coef = Nmat;

FittedData(:,1)=DataIn(:,1);
FittedData(:,2:size(Yfit,2)+1)=Yfit;
end
