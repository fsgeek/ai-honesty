#!/usr/bin/env bash
# =============================================================================
# Quick Provenance Check
# =============================================================================
#
# Lightweight pre-commit check: runs ONLY provenance_judge.py in
# paper-to-data mode. This is the "did the numbers change?" check.
#
# Skips the code-to-paper pass entirely (that audit is slower and does
# not need to run on every commit -- only when experiment code changes).
#
# Usage:
#   ./scripts/quick_provenance_check.sh          # normal run
#   ./scripts/quick_provenance_check.sh --dry-run # preview without API calls
#
# Pre-commit hook (add to .git/hooks/pre-commit or .pre-commit-config.yaml):
#   #!/bin/bash
#   ./scripts/quick_provenance_check.sh
#
# Exit codes:
#   0 - No discrepancies found (or dry run)
#   1 - Discrepancies or inconsistencies detected
#   2 - Setup error (missing API key, missing files)
#
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/home/tony/projects/ai-honesty"
PAPER_DIR="papers/sosp"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Parse arguments
DRY_RUN=""
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --quiet|-q) QUIET=1 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--quiet]"
            echo ""
            echo "Quick provenance check: paper-to-data pass only."
            echo "Verifies that numbers in the paper match the CSV data."
            echo ""
            echo "Options:"
            echo "  --dry-run   Preview configuration without making API calls"
            echo "  --quiet     Suppress progress output, only show result line"
            echo "  --help      Show this help message"
            echo ""
            echo "Exit codes:"
            echo "  0 - Clean (no discrepancies)"
            echo "  1 - Discrepancies found"
            echo "  2 - Setup error"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 2
            ;;
    esac
done

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Load .env if it exists
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "${PROJECT_ROOT}/.env"
    set +a
fi

# Check for API key (skip check on dry run)
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${DRY_RUN}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo "Set it in your environment or in ${PROJECT_ROOT}/.env"
    exit 2
fi

# Activate virtual environment
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Check that paper directory exists
if [ ! -d "${PROJECT_ROOT}/${PAPER_DIR}" ]; then
    echo "ERROR: Paper directory not found: ${PAPER_DIR}"
    exit 2
fi

# ---------------------------------------------------------------------------
# Output setup
# ---------------------------------------------------------------------------

# Use a temporary directory for quick check output to avoid cluttering
# the main reviews directory. The output is ephemeral -- if the check
# passes, nobody needs to look at it.
QUICK_DIR=$(mktemp -d "${PROJECT_ROOT}/reviews/.quick_XXXXXX")
trap 'rm -rf "${QUICK_DIR}"' EXIT

# ---------------------------------------------------------------------------
# Run paper-to-data pass only
# ---------------------------------------------------------------------------

if [ $QUIET -eq 0 ]; then
    echo "Quick provenance check: ${TIMESTAMP}"
    echo "  Mode: paper-to-data only"
    echo ""
fi

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/provenance_judge.py" \
    --paper-dir "${PAPER_DIR}" \
    --data-dir "${PROJECT_ROOT}" \
    --pass paper-to-data \
    --output-dir "${QUICK_DIR}" \
    ${DRY_RUN}

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "QUICK CHECK FAILED: provenance_judge.py exited with code ${EXIT_CODE}"
    exit 2
fi

# ---------------------------------------------------------------------------
# Analyze results
# ---------------------------------------------------------------------------

if [ -n "${DRY_RUN}" ]; then
    echo ""
    echo "Quick provenance check: dry run complete"
    exit 0
fi

OUTPUT_FILE=$(ls -t "${QUICK_DIR}"/provenance_*.jsonl 2>/dev/null | head -1)

if [ -z "${OUTPUT_FILE}" ] || [ ! -f "${OUTPUT_FILE}" ]; then
    echo "WARNING: No output file produced."
    exit 2
fi

# Count findings
TOTAL=$(grep -c '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${OUTPUT_FILE}" 2>/dev/null || echo 0)
HIGH=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${OUTPUT_FILE}" 2>/dev/null | grep -c '"severity"[[:space:]]*:[[:space:]]*"HIGH"' || echo 0)
DISCREPANCY=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${OUTPUT_FILE}" 2>/dev/null | grep -c '"status"[[:space:]]*:[[:space:]]*"DISCREPANCY"' || echo 0)
VERIFIED=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${OUTPUT_FILE}" 2>/dev/null | grep -c '"status"[[:space:]]*:[[:space:]]*"VERIFIED"' || echo 0)
UNVERIFIED=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${OUTPUT_FILE}" 2>/dev/null | grep -c '"status"[[:space:]]*:[[:space:]]*"UNVERIFIED"' || echo 0)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "QUICK PROVENANCE CHECK RESULT"
echo "========================================================================"
echo "  Claims checked:  ${TOTAL}"
echo "  VERIFIED:        ${VERIFIED}"
echo "  UNVERIFIED:      ${UNVERIFIED}"
echo "  DISCREPANCY:     ${DISCREPANCY}"
echo "  HIGH severity:   ${HIGH}"
echo ""

if [ "${DISCREPANCY}" -gt 0 ] || [ "${HIGH}" -gt 0 ]; then
    # Show the discrepancies
    if [ "${DISCREPANCY}" -gt 0 ]; then
        echo "DISCREPANCIES:"
        python3 -c "
import json
for line in open('${OUTPUT_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'finding' and rec.get('status') == 'DISCREPANCY':
        claim = rec.get('paper_claim', '?')
        if len(claim) > 100:
            claim = claim[:100] + '...'
        sev = rec.get('severity', '?')
        ev = rec.get('evidence', '')
        if len(ev) > 100:
            ev = ev[:100] + '...'
        print(f'  [{sev}] {claim}')
        print(f'         {ev}')
" 2>/dev/null
        echo ""
    fi

    if [ "${HIGH}" -gt 0 ]; then
        echo "HIGH SEVERITY FINDINGS:"
        python3 -c "
import json
for line in open('${OUTPUT_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'finding' and rec.get('severity') == 'HIGH':
        claim = rec.get('paper_claim', '?')
        if len(claim) > 100:
            claim = claim[:100] + '...'
        status = rec.get('status', '?')
        ev = rec.get('evidence', '')
        if len(ev) > 100:
            ev = ev[:100] + '...'
        print(f'  [{status}] {claim}')
        print(f'         {ev}')
" 2>/dev/null
        echo ""
    fi

    echo "Quick provenance check: FAILED (${DISCREPANCY} discrepancies, ${HIGH} high severity)"

    # Copy the output to a persistent location for review
    PERSIST_DIR="${PROJECT_ROOT}/reviews/$(date -u +%Y-%m-%d)"
    mkdir -p "${PERSIST_DIR}"
    cp "${OUTPUT_FILE}" "${PERSIST_DIR}/quick_provenance_$(date -u +%H%M%S).jsonl"
    echo "  Output saved to: ${PERSIST_DIR}/"

    exit 1
else
    echo "Quick provenance check: PASSED (${VERIFIED} verified, ${UNVERIFIED} unverified, 0 discrepancies)"
    exit 0
fi
