-- Simple supervised learning on MNIST, using a feedforward neural net (brute forcing)
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

require 'os'
require 'nn'
require 'csvigo'

local path = "./data/"
local train_file = "h2o_train.csv"
local test_file = "h2o_test.csv"

if ~os.execute("ls data/"..train_file) then
    os.execute("gzip -d "..path..train_file..".gz")
    os.execute("gzip -d "..path..test_file..".gz")
end

train = csvigo.load(path
