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


