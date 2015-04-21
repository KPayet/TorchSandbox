-- Simple supervised learning on MNIST, using a feedforward neural net (brute forcing)
-- Uses the h2o training version of mnist
-- This is a one file script, no options, just to keep it simple

-- Loading data

path = "./data/"
train_file = "h2o_train.csv.gz"
test_file = "h2o_test.csv.gz"

os.execute("gzip -d "..path..trainfile)
os.execute("gzip -d "..path..testfile)
