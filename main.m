clear all;
clc;

%% Init variable
trainingSet = readmatrix("training-set.csv");
testingSet = readmatrix("test-set-4.csv");

nInput = 3;
nReservoirNeurons = 500;
k=0.01;

wInput = normrnd(0, 0.002, [nReservoirNeurons,nInput]);
wReservoir = normrnd(0, 2/nReservoirNeurons, [nReservoirNeurons,nReservoirNeurons]);

trainingTimeStep = size(trainingSet,2);
testingTimeStep = size(testingSet,2);
predictTimeStep = 500;

rt = zeros(nReservoirNeurons,1); %r(0) = 0
R = zeros(nReservoirNeurons,trainingTimeStep);

%% Training Reservoir
for time = 1:trainingTimeStep
    term1 = wReservoir * rt;
    term2 = wInput * trainingSet(:,time);
    rt = tanh(term1 + term2);
    R(:,time) = rt;
end

%ridge regression
wOutput = trainingSet * R' * (R*R' + k*eye(nReservoirNeurons))^-1;

%feed test data
rty = zeros(nReservoirNeurons,1);

for time = 1:testingTimeStep
    term1 = wReservoir * rty;
    term2 = wInput * testingSet(:,time);
    rty = tanh(term1 + term2);
end

%predict output
rtz = rty;
Out = zeros(3,predictTimeStep);
output = wOutput * rtz;
for time = 1:predictTimeStep
    term1 = wReservoir * rtz;
    term2 = wInput * output;
    rtz = tanh(term1 + term2);
    output = wOutput * rtz;
    Out(:,time) = output;
end
yPred = Out(2,:);
writematrix(yPred,'prediction.csv') 
% plot3(Out(1,:),Out(2,:),Out(3,:))
% plot3(trainingSet(1,:),trainingSet(2,:),traininSet(3,:))

