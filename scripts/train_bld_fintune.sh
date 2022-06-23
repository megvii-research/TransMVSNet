# run this script in the root path of TransMVSNet
MVS_TRAINING="/data/BlendedMVS/"  # path to BlendedMVS dataset
CKPT="checkpoints/model_dtu.ckpt" # path to checkpoint
LOG_DIR="outputs/bld_finetune"
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi


NGPUS=8
BATCH_SIZE=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS finetune.py \
--logdir=$LOG_DIR \
--dataset=bld_train \
--trainpath=$MVS_TRAINING \
--ndepths="48,32,8"  \
--depth_inter_r="4,1,0.5" \
--dlossw="1.0,1.0,1.0" \
--loadckpt=$CKPT \
--eval_freq=1 \
--wd=0.0001 \
--nviews=4 \
--batch_size=$BATCH_SIZE \
--lr=0.0002 \
--lrepochs="6,10,14:2" \
--epochs=16 \
--trainlist=lists/bld/training_list.txt \
--testlist=lists/bld/validation_list.txt \
--numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt
