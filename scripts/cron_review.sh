#!/bin/bash
# Daily review pipeline heartbeat.
# Install: crontab -e, then add:
#   0 6 * * * /home/tony/projects/ai-honesty/scripts/cron_review.sh >> /home/tony/projects/ai-honesty/reviews/cron.log 2>&1
#
# Runs at 6 AM UTC daily. Captures paper state, runs 3 reviewers + 2 scourers.

set -euo pipefail

PROJECT_ROOT="/home/tony/projects/ai-honesty"
cd "$PROJECT_ROOT"

echo "=== Review heartbeat: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Activate environment
source .venv/bin/activate

# Run legacy pipeline (reviewers + scourers)
PYTHONUNBUFFERED=1 python scripts/daily_review_pipeline.py --reviewers 3 --scourers 2

# Run provenance pipeline (provenance judge + redundancy + conciseness)
echo "--- Provenance pipeline ---"
./scripts/run_review_pipeline.sh

echo "=== Heartbeat complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""
