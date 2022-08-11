#root
source /cvmfs/sft.cern.ch/lcg/releases/gcc/7.3.0-cb1ee/x86_64-centos7/setup.sh
source /cvmfs/sft.cern.ch/lcg/releases/ROOT/6.12.04-abd9a/x86_64-centos7-gcc7-opt/bin/thisroot.sh
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/pythia6/429.2-63d8b/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/tbb/2018_U1-d3621/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/GSL/2.1-36ee5/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
#genie
export GENIE=/n/home05/agarciasoto/software/genie_official
export LHAPATH=$GENIE/data/evgen/pdfs
export LD_LIBRARY_PATH=/n/home05/agarciasoto/software/lhapdf6/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/n/home05/agarciasoto/software/apfel/install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/log4cpp/2.8.3-aeffd/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/releases/libxml2/2.9.7-830a9/x86_64-centos7-gcc7-opt/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$GENIE/lib:$LD_LIBRARY_PATH
export PATH=$GENIE/bin:$PATH
#tauola
export TAUOLA=/n/home05/agarciasoto/software/tauola
export LD_LIBRARY_PATH=$TAUOLA/lib:$LD_LIBRARY_PATH
#proposal
export PROPOSAL=/n/home05/agarciasoto/software/proposal/install
export LD_LIBRARY_PATH=$PROPOSAL/lib64:$LD_LIBRARY_PATH
#nupropearth
export NUPROPEARTH=/n/home05/agarciasoto/software/nupropearth
export PATH=$NUPROPEARTH/bin:$PATH
export LD_LIBRARY_PATH=$NUPROPEARTH/lib:$LD_LIBRARY_PATH
export GXMLPATH=$NUPROPEARTH
