#!/usr/bin/env bash
# Overnight comparison queue: PPO (Adam / SGD) vs ARS, OpenAI-ES, PGPE, CMA-ES.
#
#   setsid nohup bash run_overnight.sh > logs/queue/queue.log 2>&1 &
#
# GPU budget: everything must be done by $DEADLINE. Training runs execute in
# priority order; a run only starts if it can finish before
# DEADLINE - EVAL_RESERVE, so later (lower-priority) runs are skipped rather
# than overrunning. Then every run is evaluated (eval_runs.py) and the
# comparison page is built (compare_runs.py).
#
# Per-run output: logs/queue/<name>.log. Status: logs/queue/status.txt.

cd "$(dirname "$0")"
DEADLINE=${DEADLINE:-$(date -d "2026-11-09 08:40" +%s)}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-60}
TRAIN_END=$((DEADLINE - EVAL_RESERVE_MIN * 60))
mkdir -p logs/queue
STATUS=logs/queue/status.txt
PY=.venv/bin/python
TASK=Mjlab-Velocity-Flat-Unitree-G1

ES_MIN=${ES_MIN:-55}
PPO_ITERS=4000      # ~25 min at 0.38 s/iteration on the RTX 5090
PPO_MIN=27

ES_COMMON="--horizon 500 --save-every 10"
PAIRS="--popsize 2048 --episodes-per-candidate 2"
CMA="--popsize 256 --episodes-per-candidate 16"
PPO_COMMON="$TASK --env.scene.num-envs 4096 --agent.max-iterations $PPO_ITERS --agent.logger tensorboard"

log() { echo "$(date '+%F %T')  $*" | tee -a "$STATUS"; }

run() {  # run <name> <minutes> <command...>
  local name=$1 minutes=$2; shift 2
  local now; now=$(date +%s)
  if (( now + (minutes + 3) * 60 > TRAIN_END )); then
    log "SKIP  $name (needs ${minutes} min, $(( (TRAIN_END - now) / 60 )) min left)"
    return
  fi
  log "START $name"
  # Hard stop slightly after the planned length, in case anything hangs.
  timeout $(( (minutes + 10) * 60 )) "$@" > "logs/queue/$name.log" 2>&1
  local code=$?
  log "END   $name (exit $code)"
}

log "queue started; training must end by $(date -d @$TRAIN_END '+%F %T'), all by $(date -d @$DEADLINE '+%F %T')"

# --- priority order -------------------------------------------------------
run ppo_adam_big        $PPO_MIN $PY train_tracked.py $PPO_COMMON --agent.run-name adam_big
run ars_linear          $ES_MIN  $PY train_es.py --algo ars --hidden $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run openai_es_small     $ES_MIN  $PY train_es.py --algo openai_es $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run pgpe_small          $ES_MIN  $PY train_es.py --algo pgpe --sigma 0.02 --lr 0.15 $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run cmaes_small         $ES_MIN  $PY train_es.py --algo cmaes $CMA --sigma 0.02 $ES_COMMON --time-limit-min $ES_MIN
run ars_small           $ES_MIN  $PY train_es.py --algo ars $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run ppo_adam_small      $PPO_MIN $PY train_tracked.py $PPO_COMMON --agent.actor.hidden-dims=32,32 --agent.run-name adam_small
run ppo_sgd_big         $PPO_MIN $PY train_tracked.py $PPO_COMMON --agent.algorithm.optimizer sgd --agent.run-name sgd_big
run ppo_sgd_big_fixedlr $PPO_MIN $PY train_tracked.py $PPO_COMMON --agent.algorithm.optimizer sgd --agent.algorithm.schedule fixed --agent.algorithm.learning-rate ${SGD_LR:-0.3} --agent.run-name sgd_big_lr${SGD_LR:-0.3}
run openai_es_small_sgd $ES_MIN  $PY train_es.py --algo openai_es --optimizer sgd $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run pgpe_small_adam     $ES_MIN  $PY train_es.py --algo pgpe --optimizer adam --sigma 0.02 $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run sep_cmaes_small     $ES_MIN  $PY train_es.py --algo sep_cmaes $CMA --sigma 0.02 $ES_COMMON --time-limit-min $ES_MIN
run openai_es_big       $ES_MIN  $PY train_es.py --algo openai_es --hidden 512 256 128 $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run ars_big             $ES_MIN  $PY train_es.py --algo ars --hidden 512 256 128 $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run pgpe_big            $ES_MIN  $PY train_es.py --algo pgpe --hidden 512 256 128 --sigma 0.02 --lr 1.0 $PAIRS $ES_COMMON --time-limit-min $ES_MIN
run sep_cmaes_big       $ES_MIN  $PY train_es.py --algo sep_cmaes --hidden 512 256 128 $CMA --sigma 0.02 $ES_COMMON --time-limit-min $ES_MIN

# --- evaluation -------------------------------------------------------------
log "START evaluation"
RUNS=$(ls -d logs/rsl_rl/g1_velocity/*/ logs/es/g1_velocity/*/ 2>/dev/null | grep -v 2026-10-31_18-05-16_tracked)
left=$(( DEADLINE - $(date +%s) - 300 ))
timeout $(( left > 60 ? left : 60 )) $PY eval_runs.py $RUNS --skip-existing > logs/queue/eval.log 2>&1
log "END   evaluation (exit $?)"
$PY compare_runs.py > logs/queue/compare.log 2>&1
log "END   comparison page (exit $?) -> logs/comparison/index.html"
log "queue finished"
