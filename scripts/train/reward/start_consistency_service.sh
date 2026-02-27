#!/bin/bash
# ==============================================================================
# Arbiter (Reward Model) Server Initialization Script
# 
# This script launches the FastAPI server hosting the Arbiter model (e.g., 32B) 
# in the background. It includes automated health checks and GPU assignment.
#
# Usage: 
#   bash start_arbiter_server.sh [GPU_ID] [PORT]
# Example: 
#   bash start_arbiter_server.sh 1 8000
# ==============================================================================

# Parse arguments with default values (GPU 1, Port 8000)
GPU_ID=${1:-1}
PORT=${2:-8000}

# Define paths
SERVER_SCRIPT_PY="train/start_consistency_model_server.py"
LOG_DIR="train/logs"
LOG_FILE="$LOG_DIR/arbiter_server_gpu${GPU_ID}_port${PORT}.log"

echo "======================================================"
echo "🚀 Initializing Arbiter Server Framework"
echo "======================================================"
echo "[Config] Target GPU   : $GPU_ID"
echo "[Config] Target Port  : $PORT"
echo "[Config] Server Script: $SERVER_SCRIPT_PY"
echo "------------------------------------------------------"

# 1. Check if the service is already running
# We use a strict regex match to ensure we don't catch unrelated python scripts
if pgrep -f "python $SERVER_SCRIPT_PY" > /dev/null; then
    echo "✅ [INFO] Arbiter model service is already running."
    exit 0
fi

# 2. Ensure log directory exists
mkdir -p "$LOG_DIR"

# 3. Launch the model service in the background using nohup
echo "⏳ [INFO] Booting Arbiter model service on GPU $GPU_ID..."
nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python $SERVER_SCRIPT_PY --port $PORT > "$LOG_FILE" 2>&1 &

# 4. Capture the Process ID
PID=$!
echo "🟢 [INFO] Service process dispatched. PID: $PID"

# 5. Health Check Polling
echo -n "⏳ [INFO] Waiting for model to load into VRAM (Timeout: 60s) "
for i in {1..30}; do
    echo -n "."
    # Ping the health endpoint (Silently fails if server is not up yet)
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo -e "\n✅ [SUCCESS] Arbiter model service is online and healthy on GPU $GPU_ID!"
        exit 0
    fi
    sleep 2 # Check every 2 seconds
done

# 6. Timeout / Failure Handling
echo -e "\n❌ [ERROR] Service initialization timed out or failed!"
echo "--- Last 20 lines of the server log ---"
tail -n 20 "$LOG_FILE"
echo "---------------------------------------"
echo "Please check the full log file for details: $LOG_FILE"
exit 1