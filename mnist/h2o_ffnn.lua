-- Simple supervised learning on MNIST, using a feedforward neural net (brute forcing)
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

require 'os'
require 'nn'
require 'csvigo'

local path = "./data/"
local train_file = "h2o_train.csv.gz"
local test_file = "h2o_test.csv.gz"

os.execute("gzip -d "..path..train_file)
os.execute("gzip -d "..path..test_file)
