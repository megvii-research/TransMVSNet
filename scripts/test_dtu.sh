#!/usr/bin/env bash
TESTPATH="/data/DTU/dtu-test" # path to dataset dtu_test
TESTLIST="lists/dtu/test.txt" 
CKPT_FILE="checkpoints/model_dtu.ckpt" # path to checkpoint file, you need to use the model_dtu.ckpt for testing
FUSIBLE_PATH="fusibile/build/Release/fusibile.exe" # path to fusible of gipuma
OUTDIR="outputs/dtu_testing" # path to output
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python test.py \
	--dataset=general_eval \
	--batch_size=1 \
	--testpath=$TESTPATH  \
	--testlist=$TESTLIST \
	--loadckpt=$CKPT_FILE \
	--outdir=$OUTDIR \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--depth_inter_r="4.0,1.0,0.5" \
	--interval_scale=1.06 \
	--filter_method="gipuma" \
	--fusibile_exe_path=$FUSIBLE_PATH | tee -a $LOG_DIR/log.txt
	#--filter_method="normal"

