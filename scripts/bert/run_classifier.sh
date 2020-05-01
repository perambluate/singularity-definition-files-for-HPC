#!/bin/bash

MPI=$1
CODE=$2
MODEL=$3
TASK=classifier
DATA=$4
NP=$5
LOG_NAME=${DATA}_${TASK}_${MODEL}_${CODE}_${MPI}.log

if [ $# != 5 ]; then
	echo "Incorrect input variables, you should input MPI CODE MODEL DATA NP in order"
else

module purge
module load mpi/$MPI
module load bert/$CODE/$MODEL/$DATA

time -p mpirun -np $NP \
#-H ops003:2,ops004:2 \
#-mca btl_tcp_if_include enp3s0f1 -npernode 2 \
#-bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
#-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
python $CODE_PATH/run_${TASK}.py \
--do_train=true \
--do_eval=false \
--do_predict=false \
--do_lower_case=false \
--task_name=$DATA \
--data_dir=$DATA_PATH \
--vocab_file=$MODEL_PATH/vocab.txt \
--bert_config_file=$MODEL_PATH/bert_config.json \
--init_checkpoint=$MODEL_PATH/bert_model.ckpt \
--output_dir=$RESULT \
--learning_rate=5e-5 \
--num_train_epochs=0.001 \
--max_seq_length=128 \
--train_batch_size=32 \
--num_accumulation_steps=1 \
--save_checkpoints_steps=1000 \
--warmup_proportion=0.1 \
--use_fp16 \
--horovod \
2>&1 | tee $LOG_NAME
fi
