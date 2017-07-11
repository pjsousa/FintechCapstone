#!/bin/bash

### FULL
# ENCODE_SIZES=(40 60 100 224)
# SAMPLE=(10 20 30 40 50 60 70 80 90 100)
# STRIDE=(3 5 7)
# BIN_SIZES=(100 50 20)

### ENCODE 224 , CHANGE BINS
# ENCODE_SIZES=(224)
# SAMPLE=(10 20 30 40 50 60 70 80 90 100)
# STRIDE=(3)
# BIN_SIZES=(100 50 20)


### BINS 50, CHANGE ENCODE (224 already done)
ENCODE_SIZES=(224)
#SAMPLE=(10 20 30 40 50 60 70 80 90 100)
SAMPLE=(20 40 50)
STRIDE=(3)
BIN_SIZES=(50)
DROPOUT=(0.0)
OPTIMIZER=("adam" "adagrad")
TRIALS=(199)

itr_subsample=-1
itr_size=-1
itr_stride=-1
itr_earlystop=-1
itr_bin=-1
itr_dropout=-1
itr_optimizer=-1
modelname=""

# TEST DIFERENT ENCODING SIZES
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
							itr_earlystop=100

							itr_subsample_float=$(bc <<<"scale=1;${itr_subsample}/100")

							modelname="FullScenarioC_TRIAL${itr_trial}_DROPOUT${itr_dropout}_OPT${itr_optimizer}_ENCODE${itr_size}_BIN${itr_bin}_STRIDE${itr_stride}_EARLYSTOP${itr_earlystop}_SAMPLE${itr_subsample}"
							weightsfilemask="FullScenarioC_TRIAL5_ENCODE224_BIN50_STRIDE3_EARLYSTOP30_SAMPLE20_0.2"
							
							weightsfilemask="FullScenarioC_TRIAL5_DROPOUT${itr_dropout}_OPT${itr_optimizer}_ENCODE224_BIN50_STRIDE3_EARLYSTOP30_SAMPLE${itr_subsample}_0${itr_subsample_float}"
							#weightsfilemask="FullScenarioC_TRIAL${itr_trial}_DROPOUT${itr_dropout}_OPT${itr_optimizer}_ENCODE224_BIN50_STRIDE3_EARLYSTOP30_SAMPLE${itr_subsample}_${itr_subsample_float}"
							

							cmd="./capstonecli --name $modelname --scenario scenarioc --fetch"
							eval $cmd
							cmd="./capstonecli --name $modelname --scenario scenarioc --fengineer"
							eval $cmd

							cmd="./capstonecli --name $modelname --scenario scenarioc --bins $itr_bin --size $itr_size --filtersize $itr_stride --subsample $itr_subsample --earlystop $itr_earlystop --finetune $weightsfilemask --dropout $itr_dropout --optimizer $itr_optimizer --train"
							eval $cmd
						done
					done
				done
			done
		done
	done
done



