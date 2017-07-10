#!/bin/bash



ENCODE_SIZES=(224)
SAMPLE=(20)
STRIDE=(3)
BIN_SIZES=(20)
TRIALS=(1 2 3 4 5)

# ENCODE_SIZES=(40 60 100)
# SAMPLE=(50)
# STRIDE=(3)
# BIN_SIZES=(XXXXXXXXX)

# ENCODE_SIZES=(40 60 100)
# SAMPLE=(50)
# STRIDE=(XXXXXXX)
# BIN_SIZES=(3333333)

itr_subsample=-1
itr_size=-1
itr_stride=-1
itr_earlystop=-1
itr_bin=-1
itr_trial=-1
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
				for itr_trial in "${TRIALS[@]}"
				do
					itr_earlystop=100

					modelname="FullBaseline_TRIAL${itr_trial}"

					cmd="./capstonecli --name $modelname --scenario baseline --fetch"
					eval $cmd
					cmd="./capstonecli --name $modelname --scenario baseline --fengineer"
					eval $cmd

					for itr_seq in $(seq 106)
					do
						cmd="./capstonecli --name $modelname --scenario baseline --train"
						eval $cmd
					done
				done
			done
		done
	done
done
