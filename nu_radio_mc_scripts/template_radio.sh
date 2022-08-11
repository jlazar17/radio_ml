#!/bin/sh                                                                                                                                                                                                   

source /n/home05/agarciasoto/software/NuRadioMC/setup.sh

INFILENAME=${INFILEDIR}/${TUNE}_s${SEED}.root

OUTNAME=$(basename $INFILENAME)

echo "python3 /n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/detsim/generator_NuPropEarth.py $SEED $MINENE $INFILENAME ${OUTDIR}/${OUTNAME}.in.hdf5"
time python3 /n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/detsim/generator_NuPropEarth.py $SEED $MINENE $INFILENAME ${OUTDIR}/${OUTNAME}.in.hdf5

echo "python3 /n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/detsim/D01detector_sim.py ${OUTDIR}/${OUTNAME}.in.hdf5 $DETECTOR $CONFIG ${OUTDIR}/${OUTNAME}.out.hdf5 ${OUTDIR}/${OUTNAME}.out.nur"
time python3 /n/home05/agarciasoto/software/NuRadioMC/analysis-scripts/gen2-tdr-2021/detsim/D01detector_sim.py ${OUTDIR}/${OUTNAME}.in.hdf5 $DETECTOR $CONFIG ${OUTDIR}/${OUTNAME}.out.hdf5 ${OUTDIR}/${OUTNAME}.out.nur

rm ${OUTDIR}/${OUTNAME}.in.hdf5
