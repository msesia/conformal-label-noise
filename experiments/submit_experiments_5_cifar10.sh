#!/bin/bash

# Parameters
CONF=101

if [[ $CONF == 101 ]]; then
  # Figure class 101
  BATCH_SIZE_LIST=(1000 2000 5000 10000)
  EPSILON_N_CLEAN_LIST=(0.1)
  EPSILON_N_CORR_LIST=(0.1)
  ESTIMATE_LIST=("none" "rho-epsilon-point")
  SEED_LIST=$(seq 1 10)

fi


# Slurm parameters
MEMO=20G                             # Memory required (20 GB)
TIME=00-01:00:00                    # Time required (1 h)
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
  for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"; do
    for EPSILON_N_CLEAN in "${EPSILON_N_CLEAN_LIST[@]}"; do
      for EPSILON_N_CORR in "${EPSILON_N_CORR_LIST[@]}"; do
        for ESTIMATE in "${ESTIMATE_LIST[@]}"; do

          JOBN="exp"$CONF"/cifar10_n"$BATCH_SIZE
          JOBN=$JOBN"_encl"$EPSILON_N_CLEAN"_enco"$EPSILON_N_CORR
          JOBN=$JOBN"_est"$ESTIMATE"_"$SEED
          OUT_FILE=$OUT_DIR"/"$JOBN".txt"
          COMPLETE=0
          if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
          fi

          if [[ $COMPLETE -eq 0 ]]; then
            # Script to be run
            SCRIPT="exp_5.sh $BATCH_SIZE $EPSILON_N_CLEAN $EPSILON_N_CORR $ESTIMATE $SEED"
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
