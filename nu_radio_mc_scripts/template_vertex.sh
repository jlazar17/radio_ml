#!/bin/sh                                                                                                                                                                                                   

source /n/home05/agarciasoto/software/setup_nupropearth.sh

GEOMETRY=${NUPROPEARTH}/src/tools/geometry-earth_icecube.root
DETPOS="0,0,6372784"  #x,y,z center cylinder position with respect Earth center [m]
DETRAD="2000"         #cylinder radius [m]
DETHGT="2700"         #cylinder height [m]
OFFSET="0."           #offset generation [m] (0=in volume)
GEOLIMIT="6478000"    #geometry limit
NEVTS="1e4"           #number of events
PROBE="12"            #input neutrino
ALPHA="1"             #energy spectrum
CTH="-1,1"            #cos zenith range
ENE="1e8,1e10"        #energy range [GeV]

OUTPUT1=${OUTDIR}/Vertex_${PREFIX}_${TUNE}_${INTYPE}_s${SEED}.root

time VertexGenerator --seed $SEED --output $OUTPUT1 --number-of-events $NEVTS --probe $PROBE --alpha $ALPHA --costheta $CTH --energy $ENE --offset $OFFSET --detector-radius $DETRAD --detector-height $DETHGT --detector-position $DETPOS --geometry-limit $GEOLIMIT --geometry $GEOMETRY --event-generator-list $INTYPE --tune $TUNE --cross-sections ${GENIE}/${XSEC}
