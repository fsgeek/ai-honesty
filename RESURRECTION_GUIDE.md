# Resurrection Guide: Exact Paths & Recovery Methods

## Current Files to Use Directly

### Alternate Intros (Already in source tree)
```
/home/tony/projects/ai-honesty/papers/sosp/intro_paxos.tex           [146 lines, HIGH VALUE]
/home/tony/projects/ai-honesty/papers/sosp/intro_composed.tex        [173 lines, HIGH VALUE]
/home/tony/projects/ai-honesty/papers/sosp/intro_previous.tex        [99 lines, INFORMATIONAL]
/home/tony/projects/ai-honesty/papers/sosp/intro_judge_generated.tex [165 lines, MODERATE]
/home/tony/projects/ai-honesty/papers/sosp/intro_restructured.tex    [141 lines, LOW]
```

### Cut Files (Preserved)
```
/home/tony/projects/ai-honesty/papers/sosp/cut_tensor_composition.tex [50 lines, MOD-HIGH VALUE]
```

### Blog Drafts (Ready to publish)
```
/home/tony/projects/ai-honesty/docs/blog-flp-epistemic.md            [118 lines, PUBLISH NOW]
/home/tony/projects/ai-honesty/docs/blog-jester-courtier.md          [DRAFT]
/home/tony/projects/ai-honesty/docs/entropy_code_observations.md     [105 lines, EXPAND & BLOG]
/home/tony/projects/ai-honesty/docs/experiment31_together_api_analysis.md [251 lines, ARXIV APPENDIX]
```

### Scour Reports & Analysis
```
/home/tony/projects/ai-honesty/docs/scour_report_20260212.md         [116 lines, PROCESS DOC]
/home/tony/projects/ai-honesty/docs/scour_report_citation_judge.md   [252 lines, ANALYSIS]
/home/tony/projects/ai-honesty/docs/scour_report_code_traces.md      [440 lines, ANALYSIS]
/home/tony/projects/ai-honesty/docs/scour_report_cross_patterns.md   [251 lines, ANALYSIS]
/home/tony/projects/ai-honesty/docs/scour_report_methodology.md      [333 lines, ANALYSIS]
/home/tony/projects/ai-honesty/docs/scour_report_traces.md           [546 lines, ANALYSIS]
```

---

## Recovering Deleted Content

### TDA Deep Dive (202 lines)
**Location:** Deleted in commit `e9871ea`
**Recovery:**
```bash
# View the deleted section
git show e9871ea^:papers/sosp/epistemic_honest.tex | head -300 | tail -250

# Or examine the full diff
git show e9871ea papers/sosp/epistemic_honest.tex | grep "^-" | head -150
```

**Key sections to restore:**
1. "Method" subsection (TDA explanation)
2. Metrics interpretation paragraph
3. Taxonomy subsection (4 categories)
4. Results with layer-wise fragmentation
5. Phase-space visualization interpretation

### Alignment Tax Case Study (95 lines + figure)
**Location:** Deleted in commit `cb8f582`
**Recovery:**
```bash
# View the removed text
git show cb8f582 papers/sosp/epistemic_honest.tex | grep "^-" | head -100

# Figure still exists (moved out of paper but in git)
git log --diff-filter=D --summary | grep mallku_tax
```

### Format-Constraint Analysis (40 lines)
**Location:** Deleted in commit `6ab8796`
**Recovery:**
```bash
# View the condensed discussion
git show 6ab8796 papers/sosp/discussion.tex | grep -A 50 "^-" | head -60
```

---

## Branch-Based Recovery

### alt_storyline Branch
**Status:** Fully preserved, contains pre-compression versions
**Location:** `remotes/origin/alt_storyline`

**To examine:**
```bash
# Compare with main
git diff main remotes/origin/alt_storyline --stat | head -30

# Check out specific files
git show remotes/origin/alt_storyline:papers/sosp/design.tex > /tmp/design_extended.tex
git show remotes/origin/alt_storyline:papers/sosp/related.tex > /tmp/related_extended.tex

# View full TDA section with figures
git show remotes/origin/alt_storyline:papers/sosp/epistemic_honest.tex | grep -A 200 "subsection{Method}" | head -250
```

**Data preserved in branch:**
- Original TDA analysis (all figures intact)
- Extended related work section
- Full methodology scour reports
- Experimental data CSVs (overhead benchmark, calibration data)

---

## Recommended Actions by Timeline

### THIS WEEK (immediate)
```bash
# Publish FLP blog post
cp docs/blog-flp-epistemic.md /tmp/blog_fpl_epistemic_FINAL.md
# → Submit to Medium or arXiv

# Review Paxos intro for workshop submission
cat papers/sosp/intro_paxos.tex | head -80  # Check opening

# Tag submission version
git tag -a v1.0-sosp-submitted -m "SOSP 2026 submission (Mar 11)" 595bdc4
```

### NEXT 2-4 WEEKS (arXiv prep)
```bash
# Recover TDA for Appendix A
git show e9871ea^:papers/sosp/epistemic_honest.tex > /tmp/epistemic_with_tda.tex
# Extract the TDA subsections

# Copy composition material for Appendix B
cp papers/sosp/cut_tensor_composition.tex papers/sosp/appendix_b_composition.tex

# Prepare Experiment 31 for Appendix C
cp docs/experiment31_together_api_analysis.md papers/sosp/appendix_c_experiment31.tex
# (Convert MD to LaTeX)
```

### 1-2 MONTHS (blog post pipeline)
```bash
# Blog 1: Paxos framing
# Source: papers/sosp/intro_paxos.tex (lines 1-90)
# Title: "Why Paxos Explains Language Model Hallucinations"

# Blog 2: Code entropy
# Source: docs/entropy_code_observations.md
# Expand with: actual AUC numbers, visualization examples
# Title: "Entropy Spikes: Where Code Models Actually Decide"

# Blog 3: Alignment tax negative result
# Source: git show cb8f582 papers/sosp/epistemic_honest.tex
# Title: "Instruction Tuning ≠ Coherence Improvements"

# Blog 4: Narrative synthesis
# Source: papers/sosp/intro_composed.tex
# Title: "How We Synthesized Four Reviewer Feedback Threads into One Intro"
```

### 3-6 MONTHS (future paper prep)
```bash
# Check alt_storyline branch for extended versions
git checkout remotes/origin/alt_storyline -- papers/sosp/design.tex
git checkout remotes/origin/alt_storyline -- papers/sosp/related.tex

# Start drafting follow-up: "Epistemic Honesty in Composed Systems"
# Core material: papers/sosp/cut_tensor_composition.tex

# Start drafting: "Cross-Architecture Epistemic Signals"
# Core material: docs/experiment31_together_api_analysis.md
```

---

## File Recovery Commands (Copy-Paste Ready)

**Get TDA methodology section:**
```bash
git show e9871ea^:papers/sosp/epistemic_honest.tex | sed -n '/\\subsection{Method}/,/\\subsection{Results}/p' > /tmp/tda_method.tex
```

**Get alignment tax analysis:**
```bash
git show cb8f582 papers/sosp/epistemic_honest.tex | grep "^-Figure\\\\ref{fig:alignmenttax}" -A 20 > /tmp/alignment_tax_text.txt
```

**Get format constraints:**
```bash
git show 6ab8796 papers/sosp/discussion.tex | grep "^-" | grep -A 30 "Format-constraint" > /tmp/format_constraints.txt
```

**Extract all intro variants for comparison:**
```bash
for intro in papers/sosp/intro_*.tex; do
  echo "=== $intro ==="
  wc -l "$intro"
done
```

---

## Validation Checklist

Before integrating any resurrected material:

- [ ] Content is self-contained (doesn't require deleted context)
- [ ] All citations are present in `papers/sosp/epistemic_honesty.bib`
- [ ] Figure references are resolvable (check `figures/` directory)
- [ ] LaTeX syntax is valid: `pdflatex -draftmode file.tex`
- [ ] No references to removed sections (e.g., "see section 5.2")
- [ ] If from alt_storyline: verify it doesn't conflict with main narrative

---

## Key Commit Hashes for Historical Reference

| Hash | Date | Topic | Impact |
|------|------|-------|--------|
| `e9871ea` | Feb 7 | TDA removal | -202 lines |
| `cb8f582` | Jan 27 | Alignment tax removal | -95 lines + fig |
| `6ab8796` | Feb 17 | Discussion compression | -100+ lines |
| `9516845` | Feb 13 | Intro revision | Intro variants created |
| `595bdc4` | Mar 11 | SUBMISSION | Final paper |

---

**Generated:** 2026-03-13
**By:** Git scour analysis
**Status:** All files preserved; ready for resurrection
