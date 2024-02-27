#!/bin/bash

# Parameters
CONF=420

if [[ $CONF == 410 ]]; then
  # Figure class 410
  DATA_LIST=("s1")
  K_LIST=(4)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.1 0.2)
  NU_LIST=(0.25 0.75)
  CONTAMINATION_LIST=("block-RR")
  EPSILON_MAX_LIST=(0.2)
  NU_MAX_LIST=(1.0)
  V_ALPHA_LIST=(0.05)
  EPSILON_N_CLEAN_LIST=(100 200 500 1000 2000 5000)
#  EPSILON_N_CLEAN_LIST=(5000)
  EPSILON_N_CORR_LIST=(1000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none" "r-e-p")
  SEED_LIST=$(seq 1 20)

elif [[ $CONF == 420 ]]; then
  # Figure class 420
  DATA_LIST=("s1")
  K_LIST=(4 8 16)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.1 0.2)
  NU_LIST=(0.0 0.25 0.5 0.75 1.0)
  CONTAMINATION_LIST=("block-RR")
  EPSILON_MAX_LIST=(0.25)
  NU_MAX_LIST=(1.0)
  V_ALPHA_LIST=(0.01)
  EPSILON_N_CLEAN_LIST=(100 200 500 1000 2000 5000 10000)
  EPSILON_N_CORR_LIST=(10000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000 100000)
  ESTIMATE_LIST=("none" "r-e-p" "r-e-ci-pb")
  SEED_LIST=$(seq 1 1)

elif [[ $CONF == 421 ]]; then
  # Figure class 420
  DATA_LIST=("s1")
  K_LIST=(4)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  NU_LIST=(0)
  CONTAMINATION_LIST=("block-RR")
  EPSILON_MAX_LIST=(0.25)
  NU_MAX_LIST=(1)
  V_ALPHA_LIST=(0.01)
  EPSILON_N_CLEAN_LIST=(10000)
  EPSILON_N_CORR_LIST=(10000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("r-e-ci-pb")
  SEED_LIST=$(seq 1 1)

elif [[ $CONF == 430 ]]; then
  # Figure class 430
  DATA_LIST=("s1")
  K_LIST=(4)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.0 0.05 0.1 0.2)
  CONTAMINATION_LIST=("block" "random")
  EPSILON_MAX_LIST=(0.2)
  NU_MAX_LIST=(1.0)
  V_ALPHA_LIST=(0.05)
  EPSILON_N_CLEAN_LIST=(100 200 500 1000 2000 5000)
  EPSILON_N_CORR_LIST=(1000)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000)
  ESTIMATE_LIST=("none" "r-e-p")
  SEED_LIST=$(seq 1 20)

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
    for K in "${K_LIST[@]}"; do
      for MODEL in "${MODEL_LIST[@]}"; do
        for EPSILON in "${EPSILON_LIST[@]}"; do
          for NU in "${NU_LIST[@]}"; do
            for CONTAMINATION in "${CONTAMINATION_LIST[@]}"; do
              for EPSILON_MAX in "${EPSILON_MAX_LIST[@]}"; do
                for NU_MAX in "${NU_MAX_LIST[@]}"; do
                  for V_ALPHA in "${V_ALPHA_LIST[@]}"; do
                    for EPSILON_N_CLEAN in "${EPSILON_N_CLEAN_LIST[@]}"; do
                      for EPSILON_N_CORR in "${EPSILON_N_CORR_LIST[@]}"; do
                        for N_TRAIN in "${N_TRAIN_LIST[@]}"; do
                          for N_CAL in "${N_CAL_LIST[@]}"; do
                            for ESTIMATE in "${ESTIMATE_LIST[@]}"; do

                              JOBN="exp"$CONF"/"$DATA"_K"$K"_"$MODEL"_e"$EPSILON"_nu"$NU"_"$CONTAMINATION
                              JOBN=$JOBN"_emax"$EPSILON_MAX"_numax"$NU_MAX"_ea"$V_ALPHA"_encl"$EPSILON_N_CLEAN"_enco"$EPSILON_N_CORR
                              JOBN=$JOBN"_nt"$N_TRAIN"_nc"$N_CAL"_est"$ESTIMATE"_"$SEED
                              OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                              #ls $OUT_FILE
                              COMPLETE=0
                              if [[ -f $OUT_FILE ]]; then
                                COMPLETE=1
                              fi

                              if [[ $COMPLETE -eq 0 ]]; then
                                # Script to be run
                                SCRIPT="exp_7.sh $CONF $DATA $K $MODEL $EPSILON $NU $CONTAMINATION $EPSILON_MAX $NU_MAX $V_ALPHA $EPSILON_N_CLEAN $EPSILON_N_CORR $N_TRAIN $N_CAL $ESTIMATE $SEED"
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
done
