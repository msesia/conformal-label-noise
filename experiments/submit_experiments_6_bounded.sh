#!/bin/bash

# Parameters
CONF=310

if [[ $CONF == 310 ]]; then
  # Figure class 310
  DATA_LIST=("s1")
  NUM_VAR_LIST=(50)
  K_LIST=(4)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  NU_LIST=(0.0 0.25 0.5 0.75 1.0)
  CONTAMINATION_LIST=("block-RR")
  EPSILON_MAX_LIST=(0.2)
  EPSILON_SE_LIST=(0.0 0.01 0.02 0.03 0.04 0.05)
  NU_MAX_LIST=(1.0)
  NU_SE_LIST=(0.0 0.02)
  V_ALPHA_LIST=(0.01)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000 100000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

elif [[ $CONF == 311 ]]; then
  # Figure class 311
  DATA_LIST=("s1")
  NUM_VAR_LIST=(50)
  K_LIST=(4)
  SIGNAL_LIST=(1.0)
  MODEL_LIST=('RFC')
  EPSILON_LIST=(0.2)
  NU_LIST=(0.0 0.25 0.5 0.75 1.0)
  CONTAMINATION_LIST=("block-RR")
  EPSILON_MAX_LIST=(0.2)
  EPSILON_SE_LIST=(0.0 0.02)
  NU_MAX_LIST=(1.0)
  NU_SE_LIST=(0.0 0.01 0.02 0.03 0.04 0.05)
  V_ALPHA_LIST=(0.01)
  N_TRAIN_LIST=(10000)
  N_CAL_LIST=(10000 100000)
  ESTIMATE_LIST=("none")
  SEED_LIST=$(seq 1 5)

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
              for NU in "${NU_LIST[@]}"; do
                for CONTAMINATION in "${CONTAMINATION_LIST[@]}"; do
                  for EPSILON_MAX in "${EPSILON_MAX_LIST[@]}"; do
                    for EPSILON_SE in "${EPSILON_SE_LIST[@]}"; do
                      for NU_MAX in "${NU_MAX_LIST[@]}"; do
                        for NU_SE in "${NU_SE_LIST[@]}"; do
                          for V_ALPHA in "${V_ALPHA_LIST[@]}"; do
                            for N_TRAIN in "${N_TRAIN_LIST[@]}"; do
                              for N_CAL in "${N_CAL_LIST[@]}"; do
                                for ESTIMATE in "${ESTIMATE_LIST[@]}"; do

                                  JOBN="exp"$CONF"/"$DATA"_p"$NUM_VAR"_K"$K"_s"$SIGNAL"_"$MODEL"_eps"$EPSILON"_nu"$NU"_"$CONTAMINATION
                                  JOBN=$JOBN"_emax"$EPSILON_MAX"_ese"$EPSILON_SE"_nmax"$NU_MAX"_nse"$NU_SE"_Va"$V_ALPHA
                                  JOBN=$JOBN"_nt"$N_TRAIN"_nc"$N_CAL"_"$ESTIMATE"_"$SEED
                                  OUT_FILE=$OUT_DIR"/"$JOBN".txt"
                                  COMPLETE=0
                                  #ls $OUT_FILE
                                  if [[ -f $OUT_FILE ]]; then
                                    COMPLETE=1
                                  fi

                                  if [[ $COMPLETE -eq 0 ]]; then
                                    # Script to be run
                                    SCRIPT="exp_6.sh $CONF $DATA $NUM_VAR $K $SIGNAL $MODEL $EPSILON $NU $CONTAMINATION $EPSILON_MAX $EPSILON_SE $NU_MAX $NU_SE $V_ALPHA $N_TRAIN $N_CAL $ESTIMATE $SEED"
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
  done
done
