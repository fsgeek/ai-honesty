# The tale build

`tale.mp4` is assembled from segments, not edited on a timeline. Each segment is
one visual plus one audio file. Change a word, re-record one file, rebuild.

## Record

1. `python build_tale.py --script` writes `recording_script.md`: one block per
   segment, with the text to read and the segment id to name the file.
2. Record each block as its own file, saved as `../audio/<session>/<id>.wav`
   (WAV; mono is fine). Leave a second or two of silence at each end — the build
   trims it, normalises loudness to -18 LUFS, and cuts any trailing mouse click.
3. Add the session to `audio_map.py`. Later entries win, so a re-recorded
   segment supersedes an earlier take without deleting anything.

**Record one file per segment.** Earlier sessions were recorded in longer blocks
and had to be split by word proportion, which misjudges a short line read slowly
("Eighteen times. She counted." is four words and eight seconds). Parts 4 and 5
are one-to-one and need no splitting.

The times in `recording_script.md` are a guide at a measured pace, not a budget.

## Build

    ../../../.venv/bin/python build_tale.py          # -> tale.mp4 + tale.srt
    ../../../.venv/bin/python build_tale.py --list   # which segments have audio
    ../../../.venv/bin/python build_tale.py --cards  # render cards only

Missing audio becomes room tone of the estimated length, so the draft always
builds. Slides come from `../epistemic_observability_pacmi26.pdf` (page numbers
in `SEGMENTS`); rebuild the deck first if a receipt slide changed. Clips come
from `../anim/*.mp4`. Illustrations come from `../png/`.

Silence is never digital zero: every gap is filled with `Background Noise.m4a`,
the recording room itself, so a pause sounds like a pause and not a dropout.

## Edit

Narration and cards live in `SEGMENTS` in `build_tale.py`, which is the single
source of truth for both the spoken text and the captions. `docs/fairy-tale-
script.md` is the narrative draft the segments came from and is **not** kept in
sync automatically.

Resolution is 1280x720 to match the clips. For 1080p: set dpi=150 in the three
clip scripts, re-render them, and run with `RES=1920x1080`.

`build/`, `../audio/`, `tale.mp4` and `recording_script.md` are untracked;
`tale.mp4` is the deliverable and `../audio/README.md` says where the takes live.
