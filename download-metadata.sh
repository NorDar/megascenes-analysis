#!/bin/bash
echo "Starting script `date `"
for index in $(seq -f "%03g" 68 458) # 0 - 458
do
    start=`date +%s`
    mkdir -p ./logs/logs-$index
    s5cmd --no-sign-request cp s3://megascenes/images/$index/*.json ./images/$index/ > ./logs/logs-$index/metadata-download.log
    end=`date +%s`
    runtime=$((end-start))
    echo "Runtime for $index is $runtime seconds"
done 

#s5cmd --no-sign-request cp s3://megascenes/images/000/*.json ./images/000/