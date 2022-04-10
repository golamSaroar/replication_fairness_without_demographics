#!/bin/bash

cd data

mkdir -p compas
curl https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv -o compas/compas-scores-two-years.csv

mkdir -p uci_adult
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -o uci_adult/adult.data
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -o uci_adult/adult.test

mkdir -p law_school
curl http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip -o law_school/LSAC_SAS.zip
unzip -o law_school/LSAC_SAS.zip -d law_school
