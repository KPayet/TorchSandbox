-- Simple supervised learning on MNIST, using a feedforward neural net (brute forcing)
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

require 'os'
require 'torch'
require 'nn'
require 'csvigo'
require 'optim'

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

-- Training -- This part is an almost copy/paste of http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_4_train

classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat('./train.log'))
testLogger = optim.Logger(paths.concat('./test.log'))

if model then
    parameters,gradParameters = model:getParameters()
end

trsize = trainData:size()
batchSize = 1

-- Training function
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData:size()) do
	 -- load new sample
	 local input = trainData.data[shuffle[i]]:double()
	 local target = trainData.labels[shuffle[i]]
	 table.insert(inputs, input)
	 table.insert(targets, target)
      end
	print("good here")
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
		       -- get new parameters
		       if x ~= parameters then
			  parameters:copy(x)
		       end

		       -- reset gradients
		       gradParameters:zero()

		       -- f is the average of all criterions
		       local f = 0

		       -- evaluate function for complete mini batch
		       for i = 1,#inputs do
			  -- estimate f
			  local output = model:forward(inputs[i])
			  local err = criterion:forward(output, targets[i])
			  f = f + err

			  -- estimate df/dW
			  local df_do = criterion:backward(output, targets[i])
			  model:backward(inputs[i], df_do)

			  -- update confusion
			  confusion:add(output, targets[i])
		       end

		       -- normalize gradients and f(X)
		       gradParameters:div(#inputs)
		       f = f/#inputs

		       -- return f and df/dX
		       return f,gradParameters
		    end

	config = config or {learningRate = 1e-3,
			 weightDecay = 0,
			 momentum = 0,
			 learningRateDecay = 5e-7}
	print("here too")
	optim.sgd(feval, parameters, config)
	print("here three")

   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:zero()

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat('./model_'..epoch..'.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

