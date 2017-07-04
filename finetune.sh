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
SAMPLE=(20)
#SAMPLE=(10)
STRIDE=(3)
BIN_SIZES=(100)
DROPOUT=(0.2 0.3 0.5)


itr_subsample=-1
itr_size=-1
itr_stride=-1
itr_earlystop=-1
itr_bin=-1
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
				itr_earlystop=30

				modelname="FullScenarioC_ENCODE${itr_size}_BIN${itr_bin}_STRIDE${itr_stride}_EARLYSTOP${itr_earlystop}_SAMPLE${itr_subsample}"
				weightsfilemask="FullScenarioC_ENCODE224_BIN100_STRIDE3_EARLYSTOP30_SAMPLE20_0.2"

				cmd="./capstonecli --name $modelname --scenario scenarioc --bins $itr_bin --size $itr_size --filtersize $itr_stride --subsample $itr_subsample --earlystop 30 --finetune $weightsfilemask --dropout 0.0 --optimizer adam --train"
				eval $cmd

			done
		done
	done
done



