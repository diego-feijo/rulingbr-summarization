#!/usr/bin/env bash
USR_DIR=.
PROBLEM=ementa_problem
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
HPARAMS=transformer_base_single_gpu
MODEL=transformer
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM

# Train
t2t-trainer \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR

# Decode
DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Vistos relatados e discutidos, acordam os minitros" >> $DECODE_FILE

BEAM_SIZE=4
ALPHA=0.6
t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
    --decode_from_file=$DECODE_FILE \
    --decode_to_file=out-ementa.txt

cat out-ementa.txt