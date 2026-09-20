"""Map Tony's recorded takes onto the build's segments.

He records along the script's natural breath, which does not match the
SEGMENTS list one-to-one: some takes cover two segments. Rather than re-cut
his audio, the build splits a take across the segments it covers, at
word-count proportion.

Three sessions exist:
  audio/               Take 1, ~3.6 words/sec -- too fast for the tale
  audio/Take 2/        the keeper, ~2.9 words/sec, recorded 19:18-19:40
                       (the one file with "Take 2" IN ITS NAME is an earlier
                       stray; the rest of the directory is the good session)

TAKES entries are (directory, file stem, [segment ids]).
"""

TAKE3 = "Take 2"      # superseded by Part 4; kept for provenance
TAKE1 = "."           # the first session, kept for segments Take 3 lacks
PART3 = "Part 3"      # the receipt-slide narration, recorded 20 Sep

PART4 = "Part 4"      # 20 Sep re-record: one file per segment, quietest floor
PART5 = "Part 5"      # the tale-narration cards, re-read at a measured pace

# Part 4 is one file per segment -- no splitting, no proportion guessing. It
# supersedes Take 1/Take 3 for every narrated segment it covers; the receipt
# narration and the summary stay in Part 3.
_PART4 = [
    "s00_intro", "s01_title", "s02_bubbles", "s03_windows", "s05_tinker", "s06_doors",
    "s08_banana", "s10_home", "s11_jester", "s12_jester_name", "s14_court", "s15_trace",
    "s16_answer", "s17_courtier", "s19_advisors_a", "s19_advisors_b", "s20_clip_gap",
    "s22_chest", "s23_clip_wombat", "s24_pulse_a", "s24_pulse_b", "s26_which_ten",
    "s27_clip_budget", "s28_words", "s31_stranger", "s32_parse", "s36_receipt_card",
]
_PART3 = [
    "s04_receipt_konishi", "s07_receipt_sweep", "s09_receipt_banana", "s13_receipt_jester",
    "s18_receipt_court", "s21_receipt_claim", "s25_receipt_interface", "s29_receipt_results",
    "s30_receipt_length", "s37_summary",
]

# The story-carrying cards, re-read slower once the recording script stopped
# under-reporting how long a measured reading takes.
_PART5 = [
    "s01_title", "s03_windows", "s05_tinker", "s08_banana",
    "s10_home", "s14_court", "s22_chest", "s36_receipt_card",
]

TAKES = ([(PART3, sid, [sid]) for sid in _PART3]
         + [(PART4, sid, [sid]) for sid in _PART4]
         + [(PART5, sid, [sid]) for sid in _PART5])

# segment id -> (directory, take stem, index within take, n segments in take)
SEG_TAKE = {}
for _dir, _stem, _segs in TAKES:
    for _i, _sid in enumerate(_segs):
        SEG_TAKE[_sid] = (_dir, _stem, _i, len(_segs))

TAKE_SEGS = {stem: segs for _, stem, segs in TAKES}
CUT_AT = {}   # no multi-segment takes remain
