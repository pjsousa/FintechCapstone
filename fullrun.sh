#!/bin/bash





## First we'll run a model that will create the encoding files in our system.
## The encodings filenames will not have the modelname which both makes it fine to use in the 
## other model further down this script, but also is important to note if we ran this same encoding dimensions with 
## a smaller subsample size, since some other images migh be already on disk.

###./capstonecli --setup

./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --fetch --fetch
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --fetch --fengineer

## This will not really encode all the data
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 0
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 1
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 2
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 100 --fencode 3

./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 50 --fencode 0
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 50 --fencode 1
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 50 --fencode 2
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 50 --fencode 3

./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 20 --fencode 0
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 20 --fencode 1
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 20 --fencode 2
./capstonecli --name DummyModelToCreateEncodingsInOurFileSystem --scenario scenarioc --subsample 1.5062 --size 40 --bins 20 --fencode 3


## This is not exactly any experiment we mention in our report. This is just an example that 
## should complete in a couple hours.
## With the 224 Encoding size this would takes ~1.5 hours per iteration, per trial.
## 
## This would be an experiment to compare the bin sizes 20,50 and 100 with an encoding size 40. 
## This experiment would be run for 5 trials. We encode the trial number in the modelname so that 
## we can then read all the <model_name>_eval_status.tmp in a notebook. Choose either the best
## or average the results (hence our reference to using bagging in our report)

#ENCODE_SIZES=(40 60 100 224)
ENCODE_SIZES=(40)
#SAMPLE=(10 20 30 40 50 60 70 80 90 100)
SAMPLE=(20)
#STRIDE=(3)
STRIDE=(3)
BIN_SIZES=(20 50 100)
DROPOUT=(0.0)
#OPTIMIZER=("adam" "adagrad" "rmsprop")
OPTIMIZER=("adam")
TRIALS=(1 2 3 4 5)

itr_subsample=-1
itr_size=-1
itr_stride=-1
itr_earlystop=-1
itr_bin=-1
itr_dropout=-1
itr_optimizer=-1
modelname=""

## This will run all the combinations of parameters we set above.
## When we mention in our report that we did a Greedy search. It was more of a manual greedy search.
## we just set up diferent scripts similar to this in which we then pinned down the best parameter from the previous
## script. But one could also run everything at once (use the commented lines above instead) and have an actual full run
## of the parameters and then present and analyze the results in any manner. Although, please be advised the 224 encodings are
## can take a long time to finish.
## This will run only until the training module.
for itr_size in "${ENCODE_SIZES[@]}"
do
	for itr_subsample in "${SAMPLE[@]}"
	do
		for itr_stride in "${STRIDE[@]}"
		do
			for itr_bin in "${BIN_SIZES[@]}"
			do
				for itr_dropout in "${DROPOUT[@]}"
				do
					for itr_optimizer in "${OPTIMIZER[@]}"
					do
						for itr_trial in "${TRIALS[@]}"
						do
							itr_earlystop=30

							itr_earlystop=30

							modelname="FullScenarioC_TRIAL${itr_trial}_DROPOUT${itr_dropout}_OPT${itr_optimizer}_ENCODE${itr_size}_BIN${itr_bin}_STRIDE${itr_stride}_EARLYSTOP${itr_earlystop}_SAMPLE${itr_subsample}"

							cmd="./capstonecli --name $modelname --scenario scenarioc --fetch"
							eval $cmd
							cmd="./capstonecli --name $modelname --scenario scenarioc --fengineer"
							eval $cmd
							cmd="./capstonecli --name $modelname --scenario scenarioc --bins $itr_bin --size $itr_size --filtersize $itr_stride --subsample $itr_subsample --earlystop $itr_earlystop --dropout $itr_dropout --optimizer $itr_optimizer --train"
							eval $cmd

						done
					done
				done
			done
		done
	done
done