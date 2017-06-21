#!/bin/bash

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 10 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_10_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_10_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_10_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_10_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_10_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_10_encode_status_df.tmp"
eval $cmd

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 20 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_20_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_20_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_20_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_20_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_20_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_20_encode_status_df.tmp"
eval $cmd

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 30 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_30_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_30_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_30_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_30_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_30_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_30_encode_status_df.tmp"
eval $cmd

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 40 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_40_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_40_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_40_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_40_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_40_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_40_encode_status_df.tmp"
eval $cmd

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 50 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_50_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_50_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_50_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_50_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_50_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_50_encode_status_df.tmp"
eval $cmd

cmd="./capstonecli --name FullScenarioC --scenario scenarioc --subsample_encode 60 --train"
eval $cmd

cmd="cp FullScenarioC_trialconfig.tmp FullScenarioC_60_trialconfig.tmp && cp FullScenarioC_fetchstatus.tmp FullScenarioC_60_fetchstatus.tmp && cp FullScenarioC_featureengineer_status.tmp FullScenarioC_60_featureengineer_status.tmp && cp FullScenarioC_train_status.tmp FullScenarioC_60_train_status.tmp && cp FullScenarioC_eval_status.tmp FullScenarioC_60_eval_status.tmp && cp FullScenarioC_encode_status_df.tmp FullScenarioC_60_encode_status_df.tmp"
eval $cmd







