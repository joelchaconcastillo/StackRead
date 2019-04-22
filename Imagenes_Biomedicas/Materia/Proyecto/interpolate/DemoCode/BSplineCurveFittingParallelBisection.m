function [BSpline,MaxError,FittedData] = BSplineCurveFittingParallelBisection...
(DataIn,Degree,CtrlError,NumOfPieceStart,MinAngle,MaxSmooth,N0ScanForDiscontinuosCase,...
GaussNewtonLoopTime,ScanKnot)
% data seperation
tic
VectorUX1 = ParallelBisection1(DataIn,Degree,CtrlError,NumOfPieceStart);
toc
tic
%% find optimal knot
[~,n]= size(VectorUX1);
OptimalKnotOut = zeros(1,n-1);
MultipleOut = zeros(1,n-1);
for ii = 1:(n-1)   
    idx00 =  VectorUX1(1,ii);
    DataKnot =  VectorUX1(1,ii+1)-VectorUX1(1,ii);    
    idx11 =  VectorUX1(2,ii+1); 
    DataInKnot = DataIn(idx00:idx11,:);
    [OptimalKnotOut(ii),MultipleOut(ii)] = TwoPieceOptimalKnotSolver1(DataInKnot,DataKnot,...
    Degree,MinAngle,MaxSmooth,N0ScanForDiscontinuosCase,GaussNewtonLoopTime,ScanKnot);
end
[BSpline,MaxError,FittedData] = BsplineFitting(OptimalKnotOut,MultipleOut,DataIn,Degree);
toc
end