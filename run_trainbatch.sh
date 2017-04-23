#!/bin/bash

for i in `seq 1 $1`
do
	cmd="python capstonecli --train"
	eval $cmd
done
