#!/bin/bash

# Parameters
CONF=43

if [[ $CONF == 41 ]]; then
  # Figure class 41
  DATA_LIST=("s1")
  NUM_VAR_LIST=(50)
  K_LIST=(2 4 8)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.0 0.05 0.1 0.2)
  CONTAMINATION_LIST=("uniform-const")
  EPSILON_MAX_LIST=(0.1)
  EPSILON_ALPHA_LIST=(0.05)
  EPSILON_TRAIN_LIST=("corrupted")
  EPSILON_N_CLEAN_LIST=(10 50 100 200 500 1000 2000 5000)
  EPSILON_N_CORR_LIST=(1000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none" "rho-epsilon-point")
  SEED_LIST=$(seq 1 20)

elif [[ $CONF == 42 ]]; then
  # Figure class 42
  DATA_LIST=("s1")
  NUM_VAR_LIST=(50)
  K_LIST=(2)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  CONTAMINATION_LIST=("uniform-const")
  EPSILON_MAX_LIST=(0.2 0.25)
  EPSILON_ALPHA_LIST=(0.001 0.01)
  EPSILON_TRAIN_LIST=("corrupted")
  EPSILON_N_CLEAN_LIST=(10 50 100 200 500 1000 2000 5000 10000)
  EPSILON_N_CORR_LIST=(1000 5000 10000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none" "rho-epsilon-point" "rho-epsilon-ci" "rho-epsilon-ci-b" "rho-epsilon-ci-pb")
  SEED_LIST=$(seq 1 5)

elif [[ $CONF == 43 ]]; then
  # Figure class 43
  DATA_LIST=("s1")
  NUM_VAR_LIST=(50)
  K_LIST=(4)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.0 0.05 0.1 0.2)
  CONTAMINATION_LIST=("block-const" "random-const" "random-enrich")
  EPSILON_MAX_LIST=(0.1)
  EPSILON_ALPHA_LIST=(0.05)
  EPSILON_TRAIN_LIST=("corrupted")
  EPSILON_N_CLEAN_LIST=(10 50 100 200 500 1000 2000 5000)
  EPSILON_N_CORR_LIST=(1000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none" "rho-epsilon-point")
  SEED_LIST=$(seq 1 20)

fi


# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-00:20:00                    # Time required (20 m)
CORE=1                              # Cores required (1)

# Assemble order prefix
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME
#ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --account=sesia_1124 --partition=main"

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
                  for EPSILON_ALPHA in "${EPSILON_ALPHA_LIST[@]}"; do
                    for EPSILON_TRAIN in "${EPSILON_TRAIN_LIST[@]}"; do
                      for EPSILON_N_CLEAN in "${EPSILON_N_CLEAN_LIST[@]}"; do
                        for EPSILON_N_CORR in "${EPSILON_N_CORR_LIST[@]}"; do
                          for N_TRAIN in "${N_TRAIN_LIST[@]}"; do
                            for N_CAL in "${N_CAL_LIST[@]}"; do
                              for ESTIMATE in "${ESTIMATE_LIST[@]}"; do

                                JOBN="exp"$CONF"/"$DATA"_p"$NUM_VAR"_K"$K"_s"$SIGNAL"_"$MODEL"_e"$EPSILON"_"$CONTAMINATION
                                JOBN=$JOBN"_emax"$EPSILON_MAX"_ea"$EPSILON_ALPHA"_"$EPSILON_TRAIN"_encl"$EPSILON_N_CLEAN"_enco"$EPSILON_N_CORR
                                JOBN=$JOBN"_nt"$N_TRAIN"_nc"$N_CAL"_est"$ESTIMATE"_"$SEED
                                OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                                COMPLETE=0
                                if [[ -f $OUT_FILE ]]; then
                                  COMPLETE=1
                                fi

                                if [[ $COMPLETE -eq 0 ]]; then
                                  # Script to be run
                                  SCRIPT="exp_4.sh $CONF $DATA $NUM_VAR $K $SIGNAL $MODEL $EPSILON $CONTAMINATION $EPSILON_MAX $EPSILON_ALPHA $EPSILON_TRAIN $EPSILON_N_CLEAN $EPSILON_N_CORR $N_TRAIN $N_CAL $ESTIMATE $SEED"
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
#                                  ./$SCRIPT
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
  done
done
