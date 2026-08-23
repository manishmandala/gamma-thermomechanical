#!/bin/bash
set -u
source /home/manishman1191/miniconda3/etc/profile.d/conda.sh
conda activate gamma
cd /home/manishman1191/ded_dt_thermomechanical_solver/examples/wall

LOGDIR=/home/manishman1191/ded_dt_thermomechanical_solver/examples/incoming_dataset/_run_logs
mkdir -p "$LOGDIR"

run_job () {
  local n=$1 tag=$2 device=$3
  local part
  part=$(printf "part%03d" "$n")
  local ds="../incoming_dataset/$part"
  local kfile="$ds/toolpath_arc_length_composition/${n}_graded_${tag}_sinusoidal.k"
  local out="$ds/toolpath_arc_length_composition/results_${tag}_sinusoidal_full"
  local log="$LOGDIR/${part}_${tag}_full.log"
  echo "[$(date)] START $part $tag on device $device" >> "$log"
  python3 run_wall.py --k "$kfile" --out "$out" --toolpath toolpath.crs \
    --input-data-dir "$ds" --stop-fraction 1.0 --n-frames 200 --device "$device" \
    >> "$log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "[$(date)] DONE $part $tag" >> "$log"
  else
    echo "[$(date)] FAILED $part $tag (exit $status)" >> "$log"
  fi
}

# GPU 0 queue: part016, part013, part010 (both modes each, sequential)
( for n in 16 13 10; do
    for tag in globalx arclength; do
      run_job "$n" "$tag" 0
    done
  done ) &
PID0=$!

# GPU 1 queue: part015, part017, part007, part008, part009
( for n in 15 17 7 8 9; do
    for tag in globalx arclength; do
      run_job "$n" "$tag" 1
    done
  done ) &
PID1=$!

# GPU 2 queue: part014, part006
( for n in 14 6; do
    for tag in globalx arclength; do
      run_job "$n" "$tag" 2
    done
  done ) &
PID2=$!

wait $PID0 $PID1 $PID2
echo "[$(date)] ALL QUEUES FINISHED" >> "$LOGDIR/_master.log"
