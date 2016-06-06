#!/bin/bash

list='assets/test_IM.csv'
input='assets/database.csv'
output='assets/test_database.csv'

echo '' > ${output}
while read line; do
    subject=$(echo $(basename ${line}) | cut -d '_' -f 1-3)
    cat ${input} | grep ${subject} >> ${output}
done < ${list}

