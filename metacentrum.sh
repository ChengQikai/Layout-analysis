 #!/bin/bash
trap 'clean_scratch' TERM EXIT
DATADIR="/storage/brno6/home/$PBS_O_LOGNAME/"

cp -r "$DATADIR/CNN" $SCRATCHDIR || exit 43
cd "$SCRATCHDIR/CNN" || exit 42

module add tensorflow-1.0.1-gpu-python3
module add numpy-1.7.1-py2.7
module add scipy-0.12.0-py2.7

python ./src/train_detector.py -b 24 -is 300 -ms -1 -o ./ -i ./split_new/train/inputs -l ./split_new/train/labels -ti ./split_new/test/inputs -tl ./split_new/test/labels || exit 44

mkdir "$DATADIR/results" || exit 45
cp -r "./06" "$DATADIR/results" || export CLEAN_SCRATCH=false 