# FintechCapstone

This is my Capstone Project for my Udacity's [Machine Learning Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009) course

The main focus of the capstone was to create a Nasdaq100 predictor. We used Deep Learning with Computer Vision architectures.

This repository is the main source for the model building.
We also built a web application the repository for which can be found [here](https://github.com/pjsousa/FintechCapstoneWeb).


## Installing our project

We used anaconda in our dev box to build and work in the capstone and we provide our environment setup in both yaml and txt for easier replication. Although, the main requirements will be:
- Tensorflow
- Keras*
- Numpy
- Pandas
- Pandas data-reader


(* Because of the difference in array ordering we will need keras to be set up using the tensorflow backend for it to work)


## Usage

The capstonecli and FintectCapstone.py hold are the main interfaces into our project. The first one makes it easier to work and experiment inside a terminal, the latter can be imported, for example, into a Jupyter Notebook.


#### capstonecli parameters

~~~
$ ./capstonecli --help
  --setup               Create directory structure.
  --dump-config DUMP_CONFIG
                        Create dump configuration file.
  --microtlist          Only use 3 tickers
  --fetch               Execute a fetch session.
  --fengineer           Execute a fengineer session.
  --train               Execute a training session.
  --evaluate            Execute an evaluation session.
  --fencode FENCODE     Perform feature encoding on the specified block.
  --name NAME           The model name.
  --scenario SCENARIO   The scenario to use.
  --subsample SUBSAMPLE
                        The subsampling probability [0 - 100]
  --size SIZE           The image side size for the mtf encoding. Used in
                        --fencode and --train
  --bins BINS           The number of bins when performing the mtf encoding.
  --filtersize FILTERSIZE
                        The filter side size for the convolutions.
  --noutputs NOUTPUTS   The number of outputs of the model.
  --FCBlocks FCBLOCKS   The number of fully connected blocks in the model.
  --arch ARCH           Copies and renames the current status files of the
                        specified model_name into an --arch name. Also copies
                        last weights files.
  --earlystop EARLYSTOP
                        Sets the tolerance for early stopping.
  --finetune FINETUNE   Filename of weights to preload the model with
  --dropout DROPOUT     Dropout probabilities to add between the fully
                        connected layers
  --optimizer OPTIMIZER
                        Optimizer to use in finetuning
  --predict             Generates a predictions file to use in the Web App
~~~

### capstonecli examples

~~~
# This would setup our project, fetch only a small subset of the companies and run a train, evaluate and predict session
# The the executions will be  stored in tmp files in the current folder. The files will the the model name in them
  ./capstonecli --setup
  ./capstonecli --name TestModelPY27 --scenario scenarioc --fetch --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --fengineer --microtlist
# The encoding can be paralelized in different windows. Or be called at once from inside a notebook
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 0 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 1 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 2 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 3 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 4 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 5 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 6 --microtlist
  ./capstonecli --name TestModelPY27 --scenario scenarioc --bins 100 --size 40 --filtersize 3 --subsample 50 --earlystop 5 --train
  ./capstonecli --name TestModelPY27 --scenario scenarioc --bins 100 --size 40 --filtersize 3 --finetune TestModelPY27_0.5  --evaluate
  ./capstonecli --name TestModelPY27 --scenario scenarioc --bins 100 --size 40 --filtersize 3 --finetune TestModelPY27_0.5  --predict
~~~

~~~
# We called baseline to our benchmark. This would perform a fetch, feature calculation and  run for a baseline setup our project. This would data for all the companies
# The the executions will be  stored in tmp files in the current folder. The files will the the model name in them
./capstonecli --name $modelname --scenario baseline --fetch
./capstonecli --name $modelname --scenario baseline --fengineer
./capstonecli --name $modelname --scenario baseline --train
~~~

