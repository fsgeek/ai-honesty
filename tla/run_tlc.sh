#!/usr/bin/env bash
# Reproduce the TLA+ results reported in the paper's appendix.
#
# Requires tla2tools.jar; set TLA_TOOLS to its path if it is not at the
# default below.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TLA_TOOLS="${TLA_TOOLS:-$HOME/projects/tlaplus/tla2tools.jar}"
if [[ ! -f "$TLA_TOOLS" ]]; then
  echo "tla2tools.jar not found at $TLA_TOOLS; set TLA_TOOLS." >&2
  exit 1
fi
tlc() { java -cp "$TLA_TOOLS" tlc2.TLC -deadlock "$@"; }

echo "=== EpistemicImpossibility: Verifiability should be VIOLATED ==="
tlc EpistemicImpossibility 2>&1 | grep -E "Error: Invariant|No error|distinct states found"

echo
echo "=== epistemic_tensor: the three positive invariants should HOLD ==="
echo "    (TensorNoWorseThanText, ResidueIsPlausibleLies, EntropyIsNotAVerdict)"
tlc epistemic_tensor 2>&1 | grep -E "Error: Invariant|No error|distinct states found"

echo
echo "=== epistemic_tensor: Verifiability should be VIOLATED ==="
echo "    The tensor narrows the gap but does not close it; the surviving"
echo "    counterexample is a fluent fabrication exported as <<Low, Coherent>>."
sed 's/^INVARIANT .*//' epistemic_tensor.cfg > /tmp/_tensor_verif.cfg
echo "INVARIANT Verifiability" >> /tmp/_tensor_verif.cfg
tlc -config /tmp/_tensor_verif.cfg epistemic_tensor 2>&1 \
  | grep -E "Error: Invariant|No error|interface_out = <<" | head -4
