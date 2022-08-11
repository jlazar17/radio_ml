#!/bin/bash                                                                                                                                                                                                 

OUTDIR=/path/to/output/dir

LOGDIR=/path/to/log/dir

PREFIX="ml"
TUNE=GHE19_00b_00_000
INTYPE=CCHEDIS
XSEC=genie_xsec/${TUNE}.xml

NJOBS=10
for SEED in $(seq 1 $NJOBS);
do
        NAME=Vtx_${PREFIX}_${TUNE}_${INTYPE}_${SEED}
        echo $NAME
        sbatch -p arguelles_delgado -t 0-01:00:00 --mem=20000 --job-name=$NAME --output ${LOGDIR}/${NAME}.log --export=PREFIX=$PREFIX,OUTDIR=$OUTDIR,SEED=$SEED,TUNE=$TUNE,INTYPE=$INTYPE,XSEC=$XSEC template_vertex.sh
done
