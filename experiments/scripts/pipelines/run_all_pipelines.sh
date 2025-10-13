#!/usr/bin/env bash

set -euo pipefail

# Usage: bash experiments/scripts/pipelines/run_all_pipelines.sh
# This script runs all available pipeline scripts in parallel

echo "Starting all WSRL pipeline experiments..."

# Function to run pipeline with error handling
run_pipeline() {
    local pipeline_script=$1
    local gpu_id=$2
    local pipeline_name=$3
    
    echo "[$(date)] Starting ${pipeline_name} on GPU ${gpu_id}"
    if bash ${pipeline_script} ${gpu_id}; then
        echo "[$(date)] ✅ ${pipeline_name} completed successfully"
    else
        echo "[$(date)] ❌ ${pipeline_name} failed"
        return 1
    fi
}

# Run pipelines in parallel (adjust GPU IDs based on available GPUs)
echo "[$(date)] Starting parallel pipeline execution..."

# Core locomotion tasks
run_pipeline "experiments/scripts/pipelines/run_locomotion_pipeline.sh" 0 "Locomotion" &
LOCOMOTION_PID=$!

# Adroit manipulation tasks  
run_pipeline "experiments/scripts/pipelines/run_adroit_pipeline.sh" 1 "Adroit" &
ADROIT_PID=$!

# AntMaze navigation tasks
run_pipeline "experiments/scripts/pipelines/run_antmaze_pipeline.sh" 2 "AntMaze" &
ANTMAZE_PID=$!

# Kitchen manipulation tasks
run_pipeline "experiments/scripts/pipelines/run_kitchen_pipeline.sh" 3 "Kitchen" &
KITCHEN_PID=$!

# Maze2D navigation tasks
run_pipeline "experiments/scripts/pipelines/run_maze2d_pipeline.sh" 4 "Maze2D" &
MAZE2D_PID=$!

# Bullet physics tasks
run_pipeline "experiments/scripts/pipelines/run_bullet_pipeline.sh" 5 "Bullet" &
BULLET_PID=$!

# Flow traffic simulation tasks
run_pipeline "experiments/scripts/pipelines/run_flow_pipeline.sh" 6 "Flow" &
FLOW_PID=$!

# CARLA autonomous driving tasks
run_pipeline "experiments/scripts/pipelines/run_carla_pipeline.sh" 7 "CARLA" &
CARLA_PID=$!

# MiniGrid tasks
run_pipeline "experiments/scripts/pipelines/run_minigrid_pipeline.sh" 8 "MiniGrid" &
MINIGRID_PID=$!

# Wait for all pipelines to complete
echo "[$(date)] Waiting for all pipelines to complete..."

wait $LOCOMOTION_PID
LOCOMOTION_STATUS=$?

wait $ADROIT_PID  
ADROIT_STATUS=$?

wait $ANTMAZE_PID
ANTMAZE_STATUS=$?

wait $KITCHEN_PID
KITCHEN_STATUS=$?

wait $MAZE2D_PID
MAZE2D_STATUS=$?

wait $BULLET_PID
BULLET_STATUS=$?

wait $FLOW_PID
FLOW_STATUS=$?

wait $CARLA_PID
CARLA_STATUS=$?

wait $MINIGRID_PID
MINIGRID_STATUS=$?

# Report results
echo ""
echo "=========================================="
echo "Pipeline Execution Summary:"
echo "=========================================="
echo "Locomotion:  $([ $LOCOMOTION_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Adroit:      $([ $ADROIT_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "AntMaze:     $([ $ANTMAZE_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Kitchen:     $([ $KITCHEN_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Maze2D:      $([ $MAZE2D_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Bullet:      $([ $BULLET_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Flow:        $([ $FLOW_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "CARLA:       $([ $CARLA_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "MiniGrid:    $([ $MINIGRID_STATUS -eq 0 ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "=========================================="

# Calculate overall success
TOTAL_PIPELINES=9
SUCCESSFUL_PIPELINES=0

[ $LOCOMOTION_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $ADROIT_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $ANTMAZE_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $KITCHEN_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $MAZE2D_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $BULLET_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $FLOW_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $CARLA_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))
[ $MINIGRID_STATUS -eq 0 ] && ((SUCCESSFUL_PIPELINES++))

echo "Overall: ${SUCCESSFUL_PIPELINES}/${TOTAL_PIPELINES} pipelines completed successfully"
echo "[$(date)] All pipeline experiments completed."

if [ $SUCCESSFUL_PIPELINES -eq $TOTAL_PIPELINES ]; then
    echo "🎉 All pipelines completed successfully!"
    exit 0
else
    echo "⚠️  Some pipelines failed. Check the logs above for details."
    exit 1
fi
