# run this script in the root path of TransMVSNet
TESTPATH="/data/TankandTemples/intermediate" # path to dataset
TESTLIST="lists/tnt/inter.txt" 												# "lists/tnt/adv.txt"
CKPT_FILE="checkpoints/model_bld.ckpt" 				   # path to checkpoint
OUTDIR="outputs/tnt_testing/inter" 								# path to save the results
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi


python test.py \
--dataset=tnt_eval \
--num_view=10 \
--batch_size=1 \
--interval_scale=1.0 \
--numdepth=192 \
--ndepths="48,32,8" \
--depth_inter_r="4,1,0.5" \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--outdir=$OUTDIR  \
--filter_method="dynamic" \
--loadckpt $CKPT_FILE ${@:2}


python dynamic_fusion.py \
--testpath=$OUTDIR \
--tntpath=$TESTPATH \
--testlist=$TESTLIST \
--outdir=$OUTDIR \
--photo_threshold=0.18 \
--thres_view=5 \
--test_dataset=tnt