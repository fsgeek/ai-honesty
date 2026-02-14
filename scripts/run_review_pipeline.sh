#!/usr/bin/env bash
# =============================================================================
# Review Pipeline Runner
# =============================================================================
#
# Runs the three review judges in sequence:
#   1. provenance_judge.py  (paper-to-data pass, then code-to-paper pass)
#   2. redundancy_judge.py  (cross-section semantic redundancy)
#   3. conciseness_judge.py (per-section prose tightening)
#
# Outputs to a timestamped directory: reviews/YYYY-MM-DD/
# Produces a summary with finding counts, diffs against previous run.
#
# Usage:
#   ./scripts/run_review_pipeline.sh            # full pipeline
#   ./scripts/run_review_pipeline.sh --dry-run  # preview without API calls
#
# Cron (daily at 6 AM UTC):
#   0 6 * * * /home/tony/projects/ai-honesty/scripts/run_review_pipeline.sh >> /home/tony/projects/ai-honesty/reviews/pipeline_cron.log 2>&1
#
# =============================================================================

set -uo pipefail
# Note: we intentionally do NOT use set -e so that one judge failure
# does not prevent the others from running.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT="/home/tony/projects/ai-honesty"
PAPER_DIR="papers/sosp"
SCRIPTS_DIR="scripts"

# Paper section files in presentation order
PAPER_FILES=(
    "${PAPER_DIR}/intro.tex"
    "${PAPER_DIR}/background.tex"
    "${PAPER_DIR}/formal_proof.tex"
    "${PAPER_DIR}/design.tex"
    "${PAPER_DIR}/eval.tex"
    "${PAPER_DIR}/discussion.tex"
    "${PAPER_DIR}/related.tex"
    "${PAPER_DIR}/conclusion.tex"
)

TODAY=$(date -u +%Y-%m-%d)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUTPUT_DIR="${PROJECT_ROOT}/reviews/${TODAY}"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

# Parse arguments
DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Runs all three review judges and produces a summary."
            echo ""
            echo "Options:"
            echo "  --dry-run   Preview configuration without making API calls"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cd "${PROJECT_ROOT}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set up logging: tee all output to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================================"
echo "REVIEW PIPELINE"
echo "========================================================================"
echo "Timestamp:  ${TIMESTAMP}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Paper dir:  ${PAPER_DIR}"
echo "Dry run:    ${DRY_RUN:-no}"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

# Load .env if it exists (for non-interactive cron runs)
if [ -f "${PROJECT_ROOT}/.env" ]; then
    # shellcheck disable=SC1091
    set -a
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ -z "$key" || "$key" == \#* ]] && continue
        # Remove leading/trailing whitespace
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # Only set if not already in environment
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "${PROJECT_ROOT}/.env"
    set +a
fi

# Check for API key
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${DRY_RUN}" ]; then
    echo ""
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo "Set it in your environment or in ${PROJECT_ROOT}/.env"
    echo "Get a key at https://openrouter.ai/keys"
    exit 1
fi

# Activate virtual environment
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
else
    echo "WARNING: Virtual environment not found at ${PROJECT_ROOT}/.venv"
    echo "Proceeding with system Python."
fi

# Verify paper files exist
MISSING_FILES=0
for f in "${PAPER_FILES[@]}"; do
    if [ ! -f "${PROJECT_ROOT}/${f}" ]; then
        echo "WARNING: Paper file not found: ${f}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ "$MISSING_FILES" -gt 0 ]; then
    echo "WARNING: ${MISSING_FILES} paper file(s) missing. Continuing with available files."
fi

# ---------------------------------------------------------------------------
# Counters for summary
# ---------------------------------------------------------------------------

PROVENANCE_EXIT=0
REDUNDANCY_EXIT=0
CONCISENESS_EXIT=0

PROVENANCE_FILE=""
REDUNDANCY_FILE=""
CONCISENESS_FILE=""

# ---------------------------------------------------------------------------
# Judge 1: Provenance (both passes)
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "JUDGE 1/3: PROVENANCE"
echo "========================================================================"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/provenance_judge.py" \
    --paper-dir "${PAPER_DIR}" \
    --data-dir "${PROJECT_ROOT}" \
    --scripts-dir "${SCRIPTS_DIR}" \
    --pass both \
    --output-dir "${OUTPUT_DIR}" \
    ${DRY_RUN}

PROVENANCE_EXIT=$?

if [ $PROVENANCE_EXIT -ne 0 ]; then
    echo ""
    echo "WARNING: Provenance judge exited with code ${PROVENANCE_EXIT}"
else
    echo ""
    echo "Provenance judge completed successfully."
fi

# Find the output file (most recent provenance_*.jsonl in output dir)
PROVENANCE_FILE=$(ls -t "${OUTPUT_DIR}"/provenance_*.jsonl 2>/dev/null | head -1)

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Judge 2: Redundancy
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "JUDGE 2/3: REDUNDANCY"
echo "========================================================================"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Build the file list from available paper files
AVAILABLE_FILES=()
for f in "${PAPER_FILES[@]}"; do
    if [ -f "${PROJECT_ROOT}/${f}" ]; then
        AVAILABLE_FILES+=("${f}")
    fi
done

if [ ${#AVAILABLE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No paper files available for redundancy analysis."
    REDUNDANCY_EXIT=1
else
    PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/redundancy_judge.py" \
        --files "${AVAILABLE_FILES[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        ${DRY_RUN}

    REDUNDANCY_EXIT=$?
fi

if [ $REDUNDANCY_EXIT -ne 0 ]; then
    echo ""
    echo "WARNING: Redundancy judge exited with code ${REDUNDANCY_EXIT}"
else
    echo ""
    echo "Redundancy judge completed successfully."
fi

REDUNDANCY_FILE=$(ls -t "${OUTPUT_DIR}"/redundancy_*.jsonl 2>/dev/null | head -1)

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Judge 3: Conciseness
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "JUDGE 3/3: CONCISENESS"
echo "========================================================================"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

if [ ${#AVAILABLE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No paper files available for conciseness analysis."
    CONCISENESS_EXIT=1
else
    PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/scripts/conciseness_judge.py" \
        --files "${AVAILABLE_FILES[@]}" \
        --output-dir "${OUTPUT_DIR}" \
        ${DRY_RUN}

    CONCISENESS_EXIT=$?
fi

if [ $CONCISENESS_EXIT -ne 0 ]; then
    echo ""
    echo "WARNING: Conciseness judge exited with code ${CONCISENESS_EXIT}"
else
    echo ""
    echo "Conciseness judge completed successfully."
fi

CONCISENESS_FILE=$(ls -t "${OUTPUT_DIR}"/conciseness_*.jsonl 2>/dev/null | head -1)

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "PIPELINE SUMMARY"
echo "========================================================================"
echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# If dry run, skip the counting
if [ -n "${DRY_RUN}" ]; then
    echo "[DRY RUN] No findings to summarize."
    echo ""
    echo "Judge exit codes:"
    echo "  Provenance:  ${PROVENANCE_EXIT}"
    echo "  Redundancy:  ${REDUNDANCY_EXIT}"
    echo "  Conciseness: ${CONCISENESS_EXIT}"
    echo ""
    echo "Review pipeline: dry run complete"
    exit 0
fi

# Count findings from each judge's JSONL output
# Each finding is a JSON line with "record_type": "finding" or similar

# --- Provenance findings ---
PROV_TOTAL=0
PROV_HIGH=0
PROV_DISCREPANCY=0
PROV_INCONSISTENT=0

if [ -n "${PROVENANCE_FILE}" ] && [ -f "${PROVENANCE_FILE}" ]; then
    # Count findings (record_type = "finding")
    PROV_TOTAL=$(grep -c '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${PROVENANCE_FILE}" 2>/dev/null || echo 0)
    PROV_HIGH=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${PROVENANCE_FILE}" 2>/dev/null | grep -c '"severity"[[:space:]]*:[[:space:]]*"HIGH"' || echo 0)
    PROV_DISCREPANCY=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${PROVENANCE_FILE}" 2>/dev/null | grep -c '"status"[[:space:]]*:[[:space:]]*"DISCREPANCY"' || echo 0)
    PROV_INCONSISTENT=$(grep '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${PROVENANCE_FILE}" 2>/dev/null | grep -c '"status"[[:space:]]*:[[:space:]]*"INCONSISTENT"' || echo 0)
    echo "Provenance judge: ${PROVENANCE_FILE##*/}"
    echo "  Findings:       ${PROV_TOTAL}"
    echo "  HIGH severity:  ${PROV_HIGH}"
    echo "  DISCREPANCY:    ${PROV_DISCREPANCY}"
    echo "  INCONSISTENT:   ${PROV_INCONSISTENT}"
else
    echo "Provenance judge: NO OUTPUT (exit code ${PROVENANCE_EXIT})"
fi

# --- Redundancy findings ---
REDUN_TOTAL=0
REDUN_HIGH=0

if [ -n "${REDUNDANCY_FILE}" ] && [ -f "${REDUNDANCY_FILE}" ]; then
    # Redundancy findings are inside the parsed.findings array in the
    # redundancy_analysis record. Extract from the summary record instead.
    REDUN_TOTAL=$(python3 -c "
import json, sys
total = 0
for line in open('${REDUNDANCY_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_findings', 0)
        break
print(total)
" 2>/dev/null || echo 0)

    REDUN_HIGH=$(python3 -c "
import json, sys
total = 0
for line in open('${REDUNDANCY_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_high', 0)
        break
print(total)
" 2>/dev/null || echo 0)

    echo ""
    echo "Redundancy judge: ${REDUNDANCY_FILE##*/}"
    echo "  Findings:       ${REDUN_TOTAL}"
    echo "  HIGH severity:  ${REDUN_HIGH}"
else
    echo ""
    echo "Redundancy judge: NO OUTPUT (exit code ${REDUNDANCY_EXIT})"
fi

# --- Conciseness findings ---
CONC_TOTAL=0
CONC_WORDS=0

if [ -n "${CONCISENESS_FILE}" ] && [ -f "${CONCISENESS_FILE}" ]; then
    CONC_TOTAL=$(python3 -c "
import json, sys
total = 0
for line in open('${CONCISENESS_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_suggestions', 0)
        break
print(total)
" 2>/dev/null || echo 0)

    CONC_WORDS=$(python3 -c "
import json, sys
total = 0
for line in open('${CONCISENESS_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_words_saved', 0)
        break
print(total)
" 2>/dev/null || echo 0)

    echo ""
    echo "Conciseness judge: ${CONCISENESS_FILE##*/}"
    echo "  Suggestions:    ${CONC_TOTAL}"
    echo "  Words saveable: ${CONC_WORDS}"
else
    echo ""
    echo "Conciseness judge: NO OUTPUT (exit code ${CONCISENESS_EXIT})"
fi

# --- Totals ---
TOTAL_FINDINGS=$((PROV_TOTAL + REDUN_TOTAL + CONC_TOTAL))
TOTAL_HIGH=$((PROV_HIGH + REDUN_HIGH))
TOTAL_PROVENANCE_ISSUES=$((PROV_DISCREPANCY + PROV_INCONSISTENT))

echo ""
echo "------------------------------------------------------------------------"
echo "TOTALS"
echo "------------------------------------------------------------------------"
echo "  Total findings:     ${TOTAL_FINDINGS}"
echo "  HIGH severity:      ${TOTAL_HIGH}"
echo "  Provenance issues:  ${TOTAL_PROVENANCE_ISSUES} (${PROV_DISCREPANCY} discrepancy + ${PROV_INCONSISTENT} inconsistent)"
echo ""

# --- Diff against previous run ---
echo "------------------------------------------------------------------------"
echo "DIFF AGAINST PREVIOUS RUN"
echo "------------------------------------------------------------------------"

# Find the most recent previous day's output directory
PREV_DIR=""
for d in $(ls -d "${PROJECT_ROOT}/reviews"/????-??-?? 2>/dev/null | sort -r); do
    dir_date=$(basename "$d")
    if [ "$dir_date" != "$TODAY" ]; then
        PREV_DIR="$d"
        break
    fi
done

if [ -n "${PREV_DIR}" ] && [ -d "${PREV_DIR}" ]; then
    echo "Previous run: ${PREV_DIR##*/}"
    echo ""

    # Compare provenance findings
    PREV_PROV_FILE=$(ls -t "${PREV_DIR}"/provenance_*.jsonl 2>/dev/null | head -1)
    if [ -n "${PREV_PROV_FILE}" ] && [ -f "${PREV_PROV_FILE}" ] && [ -n "${PROVENANCE_FILE}" ] && [ -f "${PROVENANCE_FILE}" ]; then
        PREV_PROV_TOTAL=$(grep -c '"record_type"[[:space:]]*:[[:space:]]*"finding"' "${PREV_PROV_FILE}" 2>/dev/null || echo 0)
        PROV_DELTA=$((PROV_TOTAL - PREV_PROV_TOTAL))
        if [ $PROV_DELTA -gt 0 ]; then
            echo "  Provenance: +${PROV_DELTA} new findings (was ${PREV_PROV_TOTAL}, now ${PROV_TOTAL})"
        elif [ $PROV_DELTA -lt 0 ]; then
            echo "  Provenance: ${PROV_DELTA} findings (was ${PREV_PROV_TOTAL}, now ${PROV_TOTAL})"
        else
            echo "  Provenance: no change (${PROV_TOTAL} findings)"
        fi
    else
        echo "  Provenance: no previous data to compare"
    fi

    # Compare redundancy findings
    PREV_REDUN_FILE=$(ls -t "${PREV_DIR}"/redundancy_*.jsonl 2>/dev/null | head -1)
    if [ -n "${PREV_REDUN_FILE}" ] && [ -f "${PREV_REDUN_FILE}" ] && [ -n "${REDUNDANCY_FILE}" ] && [ -f "${REDUNDANCY_FILE}" ]; then
        PREV_REDUN_TOTAL=$(python3 -c "
import json
total = 0
for line in open('${PREV_REDUN_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_findings', 0)
        break
print(total)
" 2>/dev/null || echo 0)
        REDUN_DELTA=$((REDUN_TOTAL - PREV_REDUN_TOTAL))
        if [ $REDUN_DELTA -gt 0 ]; then
            echo "  Redundancy: +${REDUN_DELTA} new findings (was ${PREV_REDUN_TOTAL}, now ${REDUN_TOTAL})"
        elif [ $REDUN_DELTA -lt 0 ]; then
            echo "  Redundancy: ${REDUN_DELTA} findings (was ${PREV_REDUN_TOTAL}, now ${REDUN_TOTAL})"
        else
            echo "  Redundancy: no change (${REDUN_TOTAL} findings)"
        fi
    else
        echo "  Redundancy: no previous data to compare"
    fi

    # Compare conciseness findings
    PREV_CONC_FILE=$(ls -t "${PREV_DIR}"/conciseness_*.jsonl 2>/dev/null | head -1)
    if [ -n "${PREV_CONC_FILE}" ] && [ -f "${PREV_CONC_FILE}" ] && [ -n "${CONCISENESS_FILE}" ] && [ -f "${CONCISENESS_FILE}" ]; then
        PREV_CONC_TOTAL=$(python3 -c "
import json
total = 0
for line in open('${PREV_CONC_FILE}'):
    rec = json.loads(line)
    if rec.get('record_type') == 'summary':
        total = rec.get('total_suggestions', 0)
        break
print(total)
" 2>/dev/null || echo 0)
        CONC_DELTA=$((CONC_TOTAL - PREV_CONC_TOTAL))
        if [ $CONC_DELTA -gt 0 ]; then
            echo "  Conciseness: +${CONC_DELTA} new suggestions (was ${PREV_CONC_TOTAL}, now ${CONC_TOTAL})"
        elif [ $CONC_DELTA -lt 0 ]; then
            echo "  Conciseness: ${CONC_DELTA} suggestions (was ${PREV_CONC_TOTAL}, now ${CONC_TOTAL})"
        else
            echo "  Conciseness: no change (${CONC_TOTAL} suggestions)"
        fi
    else
        echo "  Conciseness: no previous data to compare"
    fi
else
    echo "No previous run found for comparison."
fi

# --- Judge exit codes ---
echo ""
echo "------------------------------------------------------------------------"
echo "EXIT CODES"
echo "------------------------------------------------------------------------"
echo "  Provenance:  ${PROVENANCE_EXIT}"
echo "  Redundancy:  ${REDUNDANCY_EXIT}"
echo "  Conciseness: ${CONCISENESS_EXIT}"

# --- Output files ---
echo ""
echo "------------------------------------------------------------------------"
echo "OUTPUT FILES"
echo "------------------------------------------------------------------------"
echo "  Log:         ${LOG_FILE}"
[ -n "${PROVENANCE_FILE}" ]  && echo "  Provenance:  ${PROVENANCE_FILE}"
[ -n "${REDUNDANCY_FILE}" ]  && echo "  Redundancy:  ${REDUNDANCY_FILE}"
[ -n "${CONCISENESS_FILE}" ] && echo "  Conciseness: ${CONCISENESS_FILE}"

# --- One-line summary for email/notification ---
echo ""
echo "========================================================================"
SUMMARY_LINE="Review pipeline: ${TOTAL_FINDINGS} findings (${TOTAL_HIGH} high severity, ${TOTAL_PROVENANCE_ISSUES} provenance issues)"
echo "${SUMMARY_LINE}"
echo "========================================================================"

# Write summary line to a separate file for easy consumption
echo "${SUMMARY_LINE}" > "${OUTPUT_DIR}/summary.txt"
echo "${TIMESTAMP}" >> "${OUTPUT_DIR}/summary.txt"

# Exit with nonzero if any judge had a HIGH severity finding
if [ "${TOTAL_HIGH}" -gt 0 ]; then
    exit 1
fi

exit 0
