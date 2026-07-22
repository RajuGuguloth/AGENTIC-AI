#!/usr/bin/env bash
# Level 5 Autonomous RAG migration script
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== Level 5 Migration ==="

# Directories
mkdir -p logs memory optimization tests evaluation observability retrieval api

# Install Level 5 dependencies
if [ -d "venv" ]; then
  source venv/bin/activate
fi
pip install -r requirements.txt -q
pip install -r requirements_level5.txt -q

# Initialize runtime config
RUNTIME_CFG="optimization/runtime_config.json"
if [ ! -f "$RUNTIME_CFG" ]; then
  cat > "$RUNTIME_CFG" <<'EOF'
{
  "min_retrieval_score": 0.35,
  "query_cache_enabled": false,
  "active_prompts": {
    "generation": "default_v1",
    "verification": "default_v1",
    "relevance": "default_v1"
  },
  "ab_test": {},
  "updated_at": null
}
EOF
  echo "Created $RUNTIME_CFG"
fi

# Touch log files
touch logs/rag_traces.jsonl logs/feedback.jsonl logs/self_healing_log.jsonl
touch optimization/optimization_history.jsonl optimization/prompt_history.jsonl
touch memory/user_profiles.jsonl

# Run unit tests
echo "Running Level 5 tests..."
python -m pytest tests/test_level5.py -v || python tests/test_level5.py

# Cron jobs (optional — prints instructions)
CRON_FILE="/tmp/rag_level5_cron.txt"
cat > "$CRON_FILE" <<EOF
# Add to crontab with: crontab -e
# Daily evaluation (6 AM)
0 6 * * * cd $ROOT && source venv/bin/activate && python evaluation/daily_eval.py >> logs/daily_eval.log 2>&1
# Self-healing (hourly)
0 * * * * cd $ROOT && source venv/bin/activate && python observability/self_healer.py >> logs/self_healer.log 2>&1
# Threshold optimization (weekly, Sunday 3 AM)
0 3 * * 0 cd $ROOT && source venv/bin/activate && python optimization/threshold_optimizer.py >> logs/threshold_optimizer.log 2>&1
# Prompt optimization check (daily 4 AM)
0 4 * * * cd $ROOT && source venv/bin/activate && python -c "import asyncio; from optimization.prompt_optimizer import optimize_prompts; asyncio.run(optimize_prompts())" >> logs/prompt_optimizer.log 2>&1
EOF

echo ""
echo "=== Migration complete ==="
echo "Start app:  python app.py"
echo "Daily eval: python evaluation/daily_eval.py"
echo "Cron jobs:  see $CRON_FILE"
echo ""
echo "Optional .env additions:"
echo "  LEVEL5_ENABLED=true"
echo "  REDIS_URL=redis://localhost:6379/0"
echo "  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/..."
