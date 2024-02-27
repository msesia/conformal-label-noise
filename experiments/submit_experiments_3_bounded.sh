#!/bin/bash

# Parameters
CONF=31

if [[ $CONF == 31 ]]; then
  # Figure class 31
  DATA_LIST=("synthetic1")
  NUM_VAR_LIST=(50)
  K_LIST=(4)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.1)
  CONTAMINATION_LIST=("uniform")
  EPSILON_MAX_LIST=(0.1)
  EPSILON_SE_LIST=(0 0.01 0.02 0.03 0.04 0.05)
  EPSILON_ALPHA_LIST=(0.01)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(1000 10000 100000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

elif [[ $CONF == 32 ]]; then
  # Figure class 32
  DATA_LIST=("synthetic1")
  NUM_VAR_LIST=(50)
  K_LIST=(4)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  CONTAMINATION_LIST=("uniform")
  EPSILON_MAX_LIST=(0.2)
  EPSILON_SE_LIST=(0 0.01 0.02 0.03 0.04 0.05)
  EPSILON_ALPHA_LIST=(0.01)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(1000 10000 100000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

elif [[ $CONF == 33 ]]; then
  # Figure class 33
  DATA_LIST=("synthetic1")
  NUM_VAR_LIST=(50)
  K_LIST=(2 4 8)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  CONTAMINATION_LIST=("uniform")
  EPSILON_MAX_LIST=(0.2)
  EPSILON_SE_LIST=(0 0.01 0.02 0.03 0.04 0.05)
  EPSILON_ALPHA_LIST=(0.01)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

elif [[ $CONF == 34 ]]; then
  # Figure class 34
  DATA_LIST=("synthetic1")
  NUM_VAR_LIST=(50)
  K_LIST=(2 4 8)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.1)
  CONTAMINATION_LIST=("uniform")
  EPSILON_MAX_LIST=(0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2)
  EPSILON_SE_LIST=(0)
  EPSILON_ALPHA_LIST=(0)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

fi


# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-00:20:00                    # Time required (20 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
#ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

# Create directory for log files
LOGS="logs"
mkdir -p $LOGS
mkdir -p $LOGS"/exp"$CONF

OUT_DIR="results"
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR"/exp"$CONF


# Loop over configurations
for SEED in $SEED_LIST; do
  for DATA in "${DATA_LIST[@]}"; do
    for NUM_VAR in "${NUM_VAR_LIST[@]}"; do
      for K in "${K_LIST[@]}"; do
        for SIGNAL in "${SIGNAL_LIST[@]}"; do
          for MODEL in "${MODEL_LIST[@]}"; do
            for EPSILON in "${EPSILON_LIST[@]}"; do
              for CONTAMINATION in "${CONTAMINATION_LIST[@]}"; do
                for EPSILON_MAX in "${EPSILON_MAX_LIST[@]}"; do
                  for EPSILON_SE in "${EPSILON_SE_LIST[@]}"; do
                    for EPSILON_ALPHA in "${EPSILON_ALPHA_LIST[@]}"; do
                      for N_TRAIN in "${N_TRAIN_LIST[@]}"; do
                        for N_CAL in "${N_CAL_LIST[@]}"; do
                          for ESTIMATE in "${ESTIMATE_LIST[@]}"; do

                            JOBN="exp"$CONF"/"$DATA"_p"$NUM_VAR"_K"$K"_signal"$SIGNAL"_"$MODEL"_eps"$EPSILON"_"$CONTAMINATION
                            JOBN=$JOBN"_epsmax"$EPSILON_MAX"_epsse"$EPSILON_SE"_epsalpha"$EPSILON_ALPHA
                            JOBN=$JOBN"_nt"$N_TRAIN"_nc"$N_CAL"_est"$ESTIMATE"_seed"$SEED
                            OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                            COMPLETE=0
                            #                    ls $OUT_FILE
                            if [[ -f $OUT_FILE ]]; then
                              COMPLETE=1
                            fi

                            if [[ $COMPLETE -eq 0 ]]; then
                              # Script to be run
                              SCRIPT="exp_3.sh $CONF $DATA $NUM_VAR $K $SIGNAL $MODEL $EPSILON $CONTAMINATION $EPSILON_MAX $EPSILON_SE $EPSILON_ALPHA $N_TRAIN $N_CAL $ESTIMATE $SEED"
                              # Define job name
                              OUTF=$LOGS"/"$JOBN".out"
                              ERRF=$LOGS"/"$JOBN".err"
                              # Assemble slurm order for this job
                              ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
                              # Print order
                              echo $ORD
                              # Submit order
                              $ORD
                              # Run command now
                              #./$SCRIPT
                            fi

                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
