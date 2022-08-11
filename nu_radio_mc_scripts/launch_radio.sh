#!/bin/bash

LOGDIR=/path/to/log/dir

TUNE=Vertex_ml_GHE19_00b_00_000_CCHEDIS
INFILEDIR=/path/to/input/dir  #where vertex files are stored
DETECTOR=/n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/detector/one_station.json
CONFIG=/n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/config/config_ARZ2020_noise.yaml
OUTDIR=/path/to/output/dir
MINENE=1e3

NJOBS=10
for SEED in $(seq 1 $NJOBS);
do
	NAME=RADIO_${TUNE}_${SEED}
	echo $NAME
	sbatch -p arguelles_delgado -t 0-10:00:00 --mem=10000 --job-name=$NAME --output ${LOGDIR}/${NAME}.log --export=SEED=$SEED,TUNE=$TUNE,INFILEDIR=$INFILEDIR,DETECTOR=$DETECTOR,CONFIG=$CONFIG,OUTDIR=$OUTDIR,MINENE=$MINENE template.sh 
done


