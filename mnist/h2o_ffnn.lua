-- Simple supervised learning on MNIST, using a feedforward neural net (brute forcing)
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

require 'os'
require 'torch'
require 'nn'
require 'csvigo'

local data_path = "./data/"
local train_file = "h2o_train.csv"
local test_file = "h2o_test.csv"

if not os.execute("ls data/"..train_file) then
    print("Uncompressing data...")
    os.execute("gzip -d "..data_path..train_file..".gz")
    os.execute("gzip -d "..data_path..test_file..".gz")
end

train = csvigo.load{path = data_path..train_file, mode="raw", header=false}
train = torch.Tensor(train)
test = csvigo.load{path = data_path..test_file, mode="raw", header=false}
test = torch.Tensor(test)

trainData = {
    
    data = train[{{},{1,784}}],
    labels = train[{{}, 785}],
    size = function() return (#trainData.data)[1] end
}
testData = {
    
    data = test[{{},{1,784}}],
    labels = test[{{}, 785}],
    size = function() return (#testData.data)[1] end
}

-- Normalize features globally

mean = trainData.data:mean()
std = trainData.data:std()

trainData.data:add(-mean)
trainData.data:div(std)

testData.data:add(-mean)
testData.data:div(std)

ninputs = (#trainData.data)[2]
nhiddens = {784, 512, 256}
noutputs = 10

-- Build model

model = nn.Sequential()

--1st layer
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, nhiddens[1]))
model:add(nn.Tanh())

--2nd layer
model:add(nn.Linear(nhiddens[1], nhiddens[2]))
model:add(nn.Tanh())

--3rd layer
model:add(nn.Linear(nhiddens[2], nhiddens[3]))
model:add(nn.Tanh())

--Output layer
model:add(nn.Linear(nhiddens[3], noutputs))
model:add(nn.LogSoftMax()) -- needed for NLL criterion

--Loss function

criterion = nn.ClassNLLCriterion()


