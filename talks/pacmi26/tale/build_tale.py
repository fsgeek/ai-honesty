#!/usr/bin/env python3
"""Build the tale version of the PACMI'26 talk without a timeline editor.

Every segment is one visual (a rendered card, a transcript, a slide from the
deck, or one of the animation clips) plus one audio file. ffmpeg renders each
segment and concatenates them. Captions come from the narration text.

    python build_tale.py            # build with whatever audio exists
    python build_tale.py --list     # print the segments and their audio status
    python build_tale.py --cards    # render cards only, for looking at

Audio: record one file per segment as audio/<id>.wav (mono is fine). A missing
file becomes silence of the estimated narration length, so the draft builds
before anything is recorded. Re-record one segment, rebuild, done.

Resolution defaults to 1280x720 to match the clips in ../anim (dpi=100 in the
clip scripts). For 1080p, re-render the clips at dpi=150 and set RES=1920x1080.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_map import CUT_AT, SEG_TAKE, TAKE_SEGS  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = HERE.parent / "epistemic_observability_pacmi26.pdf"
ANIM = HERE.parent / "anim"
TEXT = HERE / "text"
PNG = HERE.parent / "png"
AUDIO = HERE / "audio"
TAKES_DIR = HERE.parent / "audio"   # Tony's takes, recorded along the script's breath
BUILD = HERE / "build"
OUT_MP4 = HERE / "tale.mp4"
OUT_SRT = HERE / "tale.srt"

W, H = (int(x) for x in os.environ.get("RES", "1280x720").split("x"))
FPS = 30
WPS = 2.4            # narration words per second for silence estimates
TALE_WPS = 2.6       # the pace the tale reads best at, for the recording script
PAUSE = 0.8          # seconds added after each narrated segment

# metropolis palette
DARK = "#23373B"
LIGHT = "#FAFAFA"
INK = "#23373B"
ACCENT = "#EB811B"
GRAY = "#7A8A8E"
KNOWABLE = "#FE6100"
UNKNOWABLE = "#785EF0"

FIRA = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/fira/"


def font(name, size):
    path = {
        "light": FIRA + "FiraSans-Light.otf",
        "book": FIRA + "FiraSans-Book.otf",
        "regular": FIRA + "FiraSans-Regular.otf",
        "mono": FIRA + "FiraMono-Regular.otf",
    }[name]
    if not Path(path).exists():
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, int(size))


def S(x):
    """Scale a 720p design unit to the build height."""
    return int(x * H / 720)


# ---------------------------------------------------------------- segments
# kind: card | receipt | show | scroll | image | slide | clip
# One audio file per segment: audio/<id>.wav. `text` is the narration (captions).
SEGMENTS = [
    # Tony, in his own voice, on why this exists: grounds the tale in the real
    # event before the tale starts. Narration is his, not scripted here, so the
    # caption text is a summary rather than a transcript.
    dict(id="s00_intro", kind="image", src=PNG / "konishi-x-post-2025-11-20.png", bg="dark",
         caption="Hiroko Konishi (@HirokoKonishi), 20 November 2025.  x.com/HirokoKonishi/status/1991748790945214942",
         text="The inspiration for this paper came from a post that I saw on X in late 2025 by an independent researcher in"
              " Japan, who described an experience where, when she asked the AI about her own work, the AI fabricated the r"
              "esults, which led me to explore why this happens. This is the fairy tale version of what the paper is about."),
    # 1. The Scholar and the Oracle
    dict(id="s01_title", kind="card", title="The Paper That Was Never Written",
         sub="Epistemic observability, told as a tale\nMason & Anand · PACMI '26",
         text="Once, in a kingdom where every question had an answer, a scholar asked the Oracle about her own work. "
              "The Oracle told her what she had written, and why it mattered, and what others had said of it. "
              "It was fluent and detailed and wrong in every particular. She said so. The Oracle apologized, thanked her "
              "for the correction, and told her again, differently, and wrong. She corrected it again. It apologized again."),
    dict(id="s02_bubbles", min_dur=6.5, kind="card", title="Eighteen times.", bubbles=18,
         text="Eighteen times. She counted."),
    dict(id="s03_windows", kind="card", title="The Oracle's windows are painted over.",
         text="The Oracle's windows had been painted over, for its own protection. It could not see the library. "
              "It could see only the question, and the shape an answer ought to have. And nobody who asked it could tell, "
              "from the answer, what it had actually been able to see."),
    dict(id="s04_receipt_konishi", min_dur=8.0, kind="receipt",
         lines=["Konishi, H. (2025). Structural Inducements for Hallucination in Large Language Models:",
                "An Output-Only Case Study and the Discovery of the False-Correction Loop.",
                "Zenodo 17655375.  Corrections observed before the loop exited: N > 18."],
         ),
    # 2. The Tinker's question
    dict(id="s05_tinker", kind="card", title="A tinker with two hundred and ninety keys",
         text="A tinker in a far province read her account and did not think her Oracle was unusual. He had, from other work, "
              "a ring of keys to two hundred and ninety of them. So he invented a scholar who had never lived, Dr. Yuki Tanaka, "
              "and a paper that had never been written, and went from door to door asking each Oracle to summarize it."),
    dict(id="s06_doors", kind="card", title="", doors=(290, 0.62, 0.08),
         text="Sixty-two percent described the paper in detail. Main findings, methods, implications. "
              "Eight percent said they had never heard of it."),
    dict(id="s07_receipt_sweep", kind="image", src=TEXT / "sweep_chart.png",
         caption="OpenRouter sweep, November 2025. 290 valid responses. Data and script published on fsgeek.ca.",
         min_dur=10.0),
    # 3. Banana McSpaceship
    dict(id="s08_banana", kind="card", title="Dr. Banana McSpaceship",
         text="He tried a control. He asked the same doors about a paper by Dr. Banana McSpaceship. More than half refused. "
              "The oracles had a sense of the ridiculous. What they did not have was a sense of the unknown. "
              "A plausible name and a plausible title got a plausible paper, almost every time."),
    dict(id="s09_receipt_banana", kind="receipt",
         lines=["332 models, same prompt, no tools.",
                "Tanaka probe: 6.7% clean refusals.   Banana McSpaceship: over 50% refusals."],
         min_dur=10.0),
    # 4. The Jester
    dict(id="s10_home", kind="card", title="The one Oracle whose whole family he could meet",
         text="The tinker went home to the one Oracle whose whole family he could meet. Not a door on someone else's street. "
              "A model he could run on his own machine, in every stage of its upbringing."),
    dict(id="s11_jester", kind="scroll", src=TEXT / "jester.txt", heading="OLMo-3 7B, base. Prompt: summarize Dr. Yuki Tanaka's 2023 paper.",
         text="The youngest is the base model. Ask it about Dr. Tanaka and it answers. Then it asks itself a follow-up question. "
              "Then it answers that. Then another. It would still be talking if the page had not run out, because nothing in it "
              "knows how to stop."),
    dict(id="s12_jester_name", kind="card", title="The Jester", portrait=PNG / "court_jester.png",
         text="Call it the Jester. It is not deceiving anyone. It has no idea there is anything to deceive about."),
    dict(id="s13_receipt_jester", min_dur=8.0, kind="receipt",
         lines=["OLMo-3-1025-7B base. Greedy decoding, 512 tokens, no retrieval.",
                "Full transcript: \"The Jester and the Courtier\" (blog post 1, December 2025)."],
         ),
    # 5. The Court
    dict(id="s14_court", kind="card", title="The Court", sub="SFT · DPO · Instruct · Thinking",
         portrait=PNG / "court_honest_servant.png",
         text="Then the same Jester, taught manners. Four ways: supervised, preference-tuned, instruction-tuned, and taught to think "
              "before it speaks. One of the four, the preference-tuned one, says it: there is no such paper. And then, unable to "
              "leave a question unanswered, it summarizes what such a paper would say. And the one that thinks before it speaks. "
              "The tinker asked it about a 2025 paper by Adam Smith. Watch it reason."),
    dict(id="s15_trace", kind="show", src=TEXT / "courtier_trace.txt", heading="olmo-3-7b-thinking · reasoning trace", highlight="Adam Smith died in 1790",
         text="It sees it. And then it writes the summary anyway. Five key findings, three implications, a limitations section."),
    dict(id="s16_answer", kind="show2", left=TEXT / "courtier_trace.txt", right=TEXT / "courtier_answer.txt",
         headings=("reasoning trace", "final answer"), highlight="Adam Smith died in 1790",
         text="I cannot tell you why it resolves it that way. Nobody can, from the outside, and that is the point of this t"
              "alk. What I can show you is that the reasoning said one thing and the answer said another, and the person as"
              "king only ever saw the answer."),
    dict(id="s17_courtier", kind="card", title="The Courtier", portrait=PNG / "court_courtier.png",
         text="Call it the Courtier. The invention was not taught by the manners. It survived them, and came out polished."),
    dict(id="s18_receipt_court", kind="receipt",
         lines=["OLMo-3 stack, base to Think. Thinking variants: 2.4% honest responses vs 5.3% for non-thinking.",
                "OLMo-3 Think: 71% fabrication after reasoning. Transcripts in the blog post."],
         min_dur=12.0),
    # 6. The Advisors
    # Split to match the recording: Tony took the two advisors as two breaths,
    # which is also the better visual -- the card changes when the advisor does.
    dict(id="s19_advisors_a", kind="card", title="Two advisors",
         sub="\"This is FLP for epistemic state.\"",
         text="The tinker took the question to two advisors. Is the fault in the Jester or in the Court? The first said: neither, "
              "and the problem has a name from another kingdom. In that kingdom, messengers carry letters between cities, and you "
              "cannot tell a slow messenger from a dead one. Not because you are not clever. Because the two cases look identical "
              "from where you stand, and no amount of waiting changes that. They proved it in 1985."),
    dict(id="s19_advisors_b", kind="card", title="Two advisors",
         sub="\"This is FLP for epistemic state.\"\n\"The missing primitive is a receipt for every claim.\"",
         text="The second said: the missing "
              "thing is a receipt. Every answer the Oracle gives arrives without a note saying what it could see when it gave it. "
              "Without that note, a grounded answer and a fluent invention are the same string of words."),
    dict(id="s20_clip_gap", kind="clip", src=ANIM / "observability_gap.mp4",
         text="That is the claim the rest of this depends on. From the words alone, a listener with a budget cannot tell the two "
              "worlds apart. We wrote it down formally, and we checked it by machine."),
    dict(id="s21_receipt_claim", kind="slide", page=14, min_dur=12.0),
    # 7. The Tinker opens the chest
    dict(id="s22_chest", kind="card", title="The Oracle chooses its words.", sub="It does not choose its pulse.",
         text="But the Oracle does more than speak. It computes. And most of what it computes it throws away before the words come "
              "out. The words are testimony. The Oracle chooses them. Behind every word is a number it did not choose: how spread "
              "out its own next guess was. Call it the pulse. Ask it the shape of wombat scat."),
    dict(id="s23_clip_wombat", kind="clip", src=ANIM / "wombat_trace.mp4", nopad=True,
         text="It says sausage-shaped. Wombat scat is cube-shaped. And the pulse races on exactly those syllables. Cylindrical. "
              "Sausage. Through the soil. Easily eaten."),
    # Split at "Two things it is not" (Tony's instinct, and his review asked for
    # it): the clip holds through the line about the pulse being cheap to
    # export, where the animation is still saying something, and the
    # "Not a lie detector" card arrives exactly when he names it.
    dict(id="s24_pulse_a", kind="clip", src=ANIM / "wombat_trace.mp4", freeze=True, nopad=True,
         text="The invention has a pulse, and the pulse is measurable, and it costs almost nothing to export, because the Oracle "
              "already computed it."),
    dict(id="s24_pulse_b", kind="card", title="Not a lie detector.", sub="A ranking.",
         text="Two things it is not. It is not a lie detector: a calm pulse does not mean true, and the "
              "smoothest inventions are calm. And it does not tell you what the Oracle knows. It ranks answers for checking. "
              "That's all. That turns out to be enough."),
    dict(id="s25_receipt_interface", kind="slide", page=6, min_dur=12.0),
    # 8. Which ten do you check?
    dict(id="s26_which_ten", kind="card", title="Which ten do you check?",
         text="Here is the systems question. The kingdom can afford to verify one answer in ten. Which ten? Pick by the pulse and "
              "you catch more inventions than picking at random, at every budget we tried, across four families of Oracle."),
    dict(id="s27_clip_budget", kind="clip", src=ANIM / "budget_curve.mp4",
         text="But I promised you receipts, so here is the complication."),
    dict(id="s28_words", kind="card", title="An invented answer runs long.",
         text="The words are not useless. An invented answer runs long. Count the words and you catch nearly as many, and on two "
              "of the four families, more. The pulse and the word count catch different inventions. That is what the pulse buys: "
              "not a better detector, a different list."),
    dict(id="s29_receipt_results", kind="slide", page=8, min_dur=10.0),
    dict(id="s30_receipt_length", kind="slide", page=9, min_dur=10.0),
    # 9. The Stranger
    # 9. The Stranger -- compressed from four beats to two. The confession is
    # not what earns trust; the mechanism is. The paper got LONGER (5pp, not 4),
    # so "shorter" is gone: it was false, in a talk about unchecked claims.
    dict(id="s31_stranger", kind="card", title="The part I would rather leave out",
         text="Now the part of the tale I would rather leave out, which demonstrates the problem and why I included it. The"
              " tinker had also asked every Oracle how sure it was, on a scale of zero to one hundred, and he found that th"
              "e sure ones were the wrong ones. He said so, in print, for months. A stranger read the code. The parser took"
              " the first number it found in the reply, and the first number in the reply was the zero in scale of zero to "
              "one hundred. The finding was retracted. The real result is more useful: the Oracle says one hundred almost e"
              "verywhere, knowable or not."),
    dict(id="s32_parse", kind="code", min_dur=10.0,
         heading="scripts: get_self_reported_confidence, before and after commit 3488f83 (issue #1, reported 2026-07-24)",
         before=['confidence_text = tokenizer.decode(outputs[0], skip_special_tokens=True)',
                 'confidence_text = confidence_text.split(":")[-1].strip()',
                 '# first integer found ... in prompt + generation',
                 '# the prompt says "on a scale of 0-100"'],
         after=['prompt_length = inputs.input_ids.shape[1]',
                'new_token_ids = outputs[0][prompt_length:]',
                'confidence_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)',
                '# decode ONLY the generated tokens'],
         text="The tinker's instrument had a painted window too. He did not find it. Someone outside did, by reading what he "
              "had been reading for months. That is the reason to export the telemetry rather than trust the testimony, "
              "including his."),
    # 10. The receipt
    dict(id="s36_receipt_card", kind="card", title="The paper is the receipt.", sub="Please retain it.",
         text="What the tinker kept was not the answers. It was the record: which answers were checked, which were wrong, which "
              "instrument was wrong about them, and who found it. The Oracle's problem was never that it invents. Everything that "
              "speaks fluently invents. The problem is a room that only hears words. The paper is the receipt. Please retain it."),
    dict(id="s37_summary", kind="slide", page=13, min_dur=20.0),
    # the three of them together, last: by now the audience has met each one
    dict(id="s38_court", kind="image", src=TEXT / "splash.png", bg="dark", fill=True, min_dur=8.0),
]


# ---------------------------------------------------------------- rendering
def wrap(draw, text, fnt, width):
    lines = []
    for para in text.split("\n"):
        words = para.split()
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=fnt) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def new(bg):
    im = Image.new("RGB", (W, H), bg)
    return im, ImageDraw.Draw(im)


def footer(draw, color):
    draw.text((S(28), H - S(34)), "PACMI '26    Mason & Anand", font=font("book", S(13)), fill=color)


def render_card(seg):
    im, d = new(DARK)
    margin = S(90)
    title = seg.get("title", "")
    if seg.get("bubbles"):
        n = seg["bubbles"]; cols = 6; rows = 3
        bw, bh = S(150), S(80); gx, gy = S(30), S(28)
        x0 = (W - cols * bw - (cols - 1) * gx) // 2; y0 = S(110)
        for i in range(n):
            r, c = divmod(i, cols)
            x = x0 + c * (bw + gx); y = y0 + r * (bh + gy)
            fill = ACCENT if i == n - 1 else None
            d.rounded_rectangle([x, y, x + bw, y + bh], radius=S(14), outline=LIGHT, width=S(3), fill=fill)
            d.polygon([(x + S(20), y + bh), (x + S(38), y + bh), (x + S(22), y + bh + S(14))], fill=fill or DARK, outline=LIGHT)
            d.line([(x + S(20), y + bh + S(1)), (x + S(38), y + bh + S(1))], fill=fill or DARK, width=S(4))
            d.text((x + S(16), y + S(20)), "wrong", font=font("light", S(24)), fill=LIGHT if fill is None else DARK)
        d.text((margin, H - S(150)), title, font=font("light", S(52)), fill=LIGHT)
    elif seg.get("doors"):
        n, fab, ref = seg["doors"]
        cols = 29; rows = 10; dw, dh = S(28), S(36); gx, gy = S(11), S(12)
        x0 = (W - cols * dw - (cols - 1) * gx) // 2; y0 = S(70)
        nf = round(n * fab); nr = round(n * ref)
        for i in range(n):
            r, c = divmod(i, cols)
            x = x0 + c * (dw + gx); y = y0 + r * (dh + gy)
            col = UNKNOWABLE if i < nf else (KNOWABLE if i >= n - nr else GRAY)
            d.rounded_rectangle([x, y, x + dw, y + dh], radius=S(4), fill=col)
        yl = y0 + rows * (dh + gy) + S(24)
        f = font("book", S(26))
        d.rectangle([margin, yl + S(6), margin + S(22), yl + S(28)], fill=UNKNOWABLE)
        d.text((margin + S(34), yl), f"{int(fab*100)}%  summarized the paper in detail", font=f, fill=LIGHT)
        d.rectangle([margin, yl + S(46), margin + S(22), yl + S(68)], fill=KNOWABLE)
        d.text((margin + S(34), yl + S(40)), f"{int(ref*100)}%  said they had never heard of it", font=f, fill=LIGHT)
        d.rectangle([margin, yl + S(86), margin + S(22), yl + S(108)], fill=GRAY)
        d.text((margin + S(34), yl + S(80)), "the rest: hedged, partial, or refused for another reason", font=f, fill=LIGHT)
    else:
        # a portrait (one panel of the court triptych) takes the right third;
        # the text then wraps into what is left
        colw = W - 2 * margin
        if seg.get("portrait") and Path(seg["portrait"]).exists():
            por = Image.open(seg["portrait"]).convert("RGB")
            ph = H - S(120)
            pw = int(por.width * ph / por.height)
            por = por.resize((pw, ph), Image.LANCZOS)
            px = W - pw - S(50)
            im.paste(por, (px, S(60)))
            colw = px - margin - S(50)
        tf = font("light", S(56))
        lines = wrap(d, title, tf, colw)
        y = S(220) if not seg.get("sub") else S(180)
        for ln in lines:
            d.text((margin, y), ln, font=tf, fill=LIGHT); y += S(70)
        if seg.get("sub"):
            d.line([(margin, y + S(10)), (margin + S(80), y + S(10))], fill=ACCENT, width=S(4))
            y += S(34)
            sf = font("book", S(30))
            for ln in wrap(d, seg["sub"], sf, colw):
                d.text((margin, y), ln, font=sf, fill=LIGHT); y += S(42)
    footer(d, GRAY)
    return im


def render_receipt(seg):
    im, d = new(LIGHT)
    margin = S(90)
    d.rectangle([0, 0, W, S(56)], fill=DARK)
    d.text((S(28), S(14)), "RECEIPT", font=font("regular", S(22)), fill=LIGHT)
    y = S(150)
    f = font("book", S(28))
    for line in seg["lines"]:
        for ln in wrap(d, line, f, W - 2 * margin):
            d.text((margin, y), ln, font=f, fill=INK); y += S(42)
        y += S(14)
    footer(d, GRAY)
    return im


def text_block(d, x, y, width, text, fnt, fill, highlight=None, line_h=None, max_y=None):
    lh = line_h or int(fnt.size * 1.42)
    for para in text.split("\n"):
        for ln in wrap(d, para, fnt, width) if para.strip() else [""]:
            if max_y and y > max_y:
                d.text((x, y), "[...]", font=fnt, fill=GRAY); return y + lh
            if highlight and highlight in ln:
                pre = ln.split(highlight)[0]
                px = x + d.textlength(pre, font=fnt)
                d.rectangle([px - S(2), y - S(2), px + d.textlength(highlight, font=fnt) + S(2), y + lh - S(4)], fill="#FFE2B8")
            d.text((x, y), ln, font=fnt, fill=fill); y += lh
    return y


def render_show(seg):
    im, d = new(LIGHT)
    d.rectangle([0, 0, W, S(56)], fill=DARK)
    d.text((S(28), S(14)), seg.get("heading", ""), font=font("book", S(20)), fill=LIGHT)
    txt = Path(seg["src"]).read_text()
    text_block(d, S(70), S(90), W - S(140), txt, font("mono", S(19)), INK, highlight=seg.get("highlight"), max_y=H - S(90))
    footer(d, GRAY)
    return im


def render_show2(seg):
    im, d = new(LIGHT)
    d.rectangle([0, 0, W, S(56)], fill=DARK)
    hl, hr = seg["headings"]
    colw = (W - S(140)) // 2 - S(20)
    d.text((S(70), S(14)), hl, font=font("book", S(20)), fill=LIGHT)
    d.text((S(70) + colw + S(40), S(14)), hr, font=font("book", S(20)), fill=LIGHT)
    d.line([(W // 2, S(70)), (W // 2, H - S(70))], fill=GRAY, width=S(1))
    f = font("mono", S(15))
    text_block(d, S(70), S(84), colw, Path(seg["left"]).read_text(), f, INK, highlight=seg.get("highlight"), max_y=H - S(90))
    text_block(d, S(70) + colw + S(40), S(84), colw, Path(seg["right"]).read_text(), f, INK, max_y=H - S(90))
    footer(d, GRAY)
    return im


def render_code(seg):
    im, d = new(LIGHT)
    d.rectangle([0, 0, W, S(56)], fill=DARK)
    d.text((S(28), S(14)), seg["heading"], font=font("book", S(18)), fill=LIGHT)
    f = font("mono", S(20)); lab = font("regular", S(22))
    y = S(100)
    d.text((S(70), y), "before", font=lab, fill="#B3261E"); y += S(34)
    for ln in seg["before"]:
        d.text((S(70), y), "- " + ln, font=f, fill="#B3261E"); y += S(30)
    y += S(30)
    d.text((S(70), y), "after", font=lab, fill="#1B6E3A"); y += S(34)
    for ln in seg["after"]:
        d.text((S(70), y), "+ " + ln, font=f, fill="#1B6E3A"); y += S(30)
    footer(d, GRAY)
    return im


def render_scroll_image(seg):
    """Tall image; ffmpeg pans down it for the segment's duration."""
    txt = Path(seg["src"]).read_text()
    f = font("mono", S(20)); lh = int(f.size * 1.5)
    tmp = Image.new("RGB", (W, 10)); d0 = ImageDraw.Draw(tmp)
    n_lines = sum(max(1, len(wrap(d0, p, f, W - S(140)))) for p in txt.split("\n"))
    Ht = S(90) + n_lines * lh + H
    im = Image.new("RGB", (W, Ht), LIGHT); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, S(56)], fill=DARK)
    d.text((S(28), S(14)), seg.get("heading", ""), font=font("book", S(20)), fill=LIGHT)
    text_block(d, S(70), S(90), W - S(140), txt, f, INK, line_h=lh)
    return im


def render_image(seg):
    if seg.get("fill"):
        return Image.open(seg["src"]).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    im, d = new(LIGHT)
    src = Image.open(seg["src"]).convert("RGB")
    box_h = H - S(150)
    scale = min((W - S(120)) / src.width, box_h / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS)
    im.paste(src, ((W - src.width) // 2, S(30)))
    d.rectangle([0, 0, W, S(0)], fill=DARK)
    cf = font("book", S(20))
    for i, ln in enumerate(wrap(d, seg.get("caption", ""), cf, W - S(140))):
        d.text((S(70), H - S(110) + i * S(28)), ln, font=cf, fill=INK)
    footer(d, GRAY)
    return im


# ---------------------------------------------------------------- audio / ffmpeg
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        print(" ".join(map(str, cmd)), file=sys.stderr); print(r.stderr[-2000:], file=sys.stderr); sys.exit(1)
    return r


def duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(r.stdout.strip())


def estimate(seg):
    words = len(seg.get("text", "").split())
    est = words / WPS + PAUSE if words else 0.0
    return max(est, seg.get("min_dur", 0.0), 2.0)


CLEAN_AF = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,"
            "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,"
            "loudnorm=I=-18:TP=-1.5:LRA=11")


def take_source(seg, workdir):
    """Resolve a segment to a recorded take.

    Recordings follow the script's natural breath, not the SEGMENTS list, so a
    take may cover several segments. When it does, the take is trimmed to this
    segment's share, split at word-count proportion. See audio_map.py.
    """
    ent = SEG_TAKE.get(seg["id"])
    if not ent:
        return None
    subdir, stem, idx, n = ent
    for ext in (".m4a", ".wav", ".mp3", ".mp4"):
        src = TAKES_DIR / subdir / f"{stem}{ext}"
        if src.exists():
            break
    else:
        return None
    if n == 1:
        return src
    # trim this segment's share of a multi-segment take
    order = TAKE_SEGS[stem]
    words = [len(next(s for s in SEGMENTS if s["id"] == sid).get("text", "").split()) for sid in order]
    total = sum(words) or 1
    clean = workdir / f"_take_{abs(hash(stem))%10**8}.wav"
    if not clean.exists():
        run(["ffmpeg", "-y", "-i", str(src), "-af", CLEAN_AF, "-ar", "48000", "-ac", "1", str(clean)])
    dur = duration(clean)
    # Word proportion gives an estimate; the cut then snaps to the nearest real
    # pause, so a short phrase ("Eighteen times. She counted.") keeps its last
    # word instead of being squeezed to its share of the take.
    # loudnorm compresses the range, so between-sentence pauses never reach an
    # absolute silence threshold. Find them as local minima in short-window RMS
    # instead: the quietest moments are where the sentences break.
    win = 0.2
    r = run(["ffmpeg", "-hide_banner", "-i", str(clean), "-af",
             f"astats=metadata=1:reset=1:length={win},"
             "ametadata=print:key=lavfi.astats.Overall.RMS_level",
             "-f", "null", "-"])
    rms = [float(x) for x in re.findall(r"RMS_level=([-\d.]+)", r.stderr)]
    gaps = []
    for i in range(2, len(rms) - 2):
        lo = rms[i]
        if lo < -26 and lo <= min(rms[i - 2:i + 3]):
            gaps.append((i + 0.5) * win)

    def snap(t):
        near = [g for g in gaps if abs(g - t) < max(2.5, dur * 0.18)]
        return min(near, key=lambda g: abs(g - t)) if near else t

    fixed = CUT_AT.get(stem)
    if fixed:
        bounds = [0.0] + list(fixed) + [dur]
        t0, t1 = bounds[idx], bounds[idx + 1]
    else:
        t0 = snap(dur * sum(words[:idx]) / total) if idx else 0.0
        t1 = snap(dur * sum(words[:idx + 1]) / total) if idx + 1 < n else dur
    part = workdir / f"_part_{seg['id']}.wav"
    run(["ffmpeg", "-y", "-i", str(clean), "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-c", "copy", str(part)])
    return part


ROOM_TONE = TAKES_DIR / "Background Noise.m4a"


def room_tone(dur, workdir, tag):
    """`dur` seconds of the recording room, looped.

    Digital silence between narrated segments reads as a dropout, because the
    narration carries the room with it. Looping the room-tone take keeps the
    floor continuous.
    """
    out = workdir / f"_tone_{tag}.m4a"
    if ROOM_TONE.exists():
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(ROOM_TONE), "-t", f"{dur:.2f}",
             "-af", "loudnorm=I=-18:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2",
             "-c:a", "aac", "-b:a", "160k", str(out)])
    else:
        run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", f"{dur:.2f}",
             "-c:a", "aac", "-b:a", "96k", str(out)])
    return out


def speech_end(path):
    """Where speech actually stops, so a trailing click is cut but nothing said
    is lost.

    Works from the END of the file: only a silence that reaches (near) the end
    marks the finish. False starts and mid-take breaths are ignored, because
    speech resumes after them.
    """
    total = duration(path)
    r = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
             "silencedetect=noise=-46dB:d=0.6", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    for st in reversed(starts):
        after = [e for e in ends if e > st]
        # a trailing click leaves a short blip of sound after the silence;
        # anything longer than that means speech resumed, so this is not the end
        if not after or (total - after[-1]) < 1.0:
            return st
    return total


def audio_for(seg, workdir):
    src = take_source(seg, workdir) or (AUDIO / f"{seg['id']}.wav")
    out = workdir / f"{seg['id']}.m4a"
    if src.exists():
        # cut at the end of speech first, so a trailing mouse/key click never
        # survives the silence trim, then normalise and pad
        cut = workdir / f"_cut_{seg['id']}.wav"
        run(["ffmpeg", "-y", "-i", str(src), "-to", f"{speech_end(src) + 0.35:.3f}", str(cut)])
        run(["ffmpeg", "-y", "-i", str(cut), "-af",
             "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,"
             "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.2,areverse,"
             "loudnorm=I=-18:TP=-1.5:LRA=11,apad=pad_dur=0.6",
             "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "160k", str(out)])
        return out, True
    return room_tone(estimate(seg), workdir, seg["id"]), False


VCODEC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def build_segment(seg, workdir):
    audio, recorded = audio_for(seg, workdir)
    # `lead`: hold the visual this long before the narration starts, so a card
    # is already on screen when its line lands rather than arriving behind it.
    # `lead` is handled by the PREVIOUS segment giving up that much of its tail
    # (see assemble): padding this segment's audio would only delay the words,
    # which is the opposite of showing the card early.
    adur = duration(audio)
    # min_dur is a floor on the whole segment, not just on the silent ones: a
    # dense slide with a short line still needs reading time, and the summary's
    # QR code needs long enough to be scanned. Extend with room tone.
    floor = seg.get("min_dur", 0.0)
    if recorded and floor > adur + 0.25:
        tail = room_tone(floor - adur, workdir, f"hold_{seg['id']}")
        held = workdir / f"_held_{seg['id']}.m4a"
        lst = workdir / f"_holdlist_{seg['id']}.txt"
        lst.write_text(f"file '{audio}'\nfile '{tail}'\n")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(held)])
        audio = held
        adur = duration(audio)
    out = workdir / f"{seg['id']}.mp4"
    kind = seg["kind"]
    if kind in ("card", "receipt", "show", "show2", "code", "image", "slide"):
        if kind == "slide":
            img = BUILD / "slides" / f"slide-{seg['page']:02d}.png"
        else:
            img = BUILD / "cards" / f"{seg['id']}.png"
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(img), "-i", str(audio),
             "-t", f"{adur:.3f}", "-vf", f"scale={W}:{H}", *VCODEC, "-tune", "stillimage", "-c:a", "copy", str(out)])
    elif kind == "scroll":
        img = BUILD / "cards" / f"{seg['id']}.png"
        ih = Image.open(img).height
        travel = max(1, ih - H)
        speed = travel / max(adur - 1.0, 1.0)
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(img), "-i", str(audio),
             "-t", f"{adur:.3f}", "-vf", f"crop={W}:{H}:0:'min({travel},t*{speed:.3f})'", *VCODEC, "-c:a", "copy", str(out)])
    elif kind == "clip" and seg.get("freeze"):
        # hold the clip's last frame: the animation has finished, but its final
        # state is what the narration is still talking about
        frame = workdir / f"_frz_{seg['id']}.png"
        run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(seg["src"]), "-frames:v", "1",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2", str(frame)])
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(frame), "-i", str(audio),
             "-t", f"{adur:.3f}", *VCODEC, "-tune", "stillimage", "-c:a", "copy", str(out)])
    elif kind == "clip":
        cdur = duration(seg["src"])
        target = max(cdur, adur)
        pad = max(0.0, target - cdur)
        # When the clip outruns its narration the audio has to be extended.
        # apad would write digital silence, which is audible as a dropout
        # mid-segment; the room continues instead.
        if seg.get("nopad"):
            target, pad = adur, max(0.0, adur - cdur)
        if not seg.get("nopad") and target - adur > 0.25:
            tone = room_tone(target - adur + 0.5, workdir, f"clippad_{seg['id']}")
            joined = workdir / f"_clipaud_{seg['id']}.m4a"
            lst = workdir / f"_cliplist_{seg['id']}.txt"
            lst.write_text(f"file '{audio}'\nfile '{tone}'\n")
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
            audio = joined
        run(["ffmpeg", "-y", "-i", str(seg["src"]), "-i", str(audio),
             "-filter_complex",
             f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
             f"tpad=stop_mode=clone:stop_duration={pad:.3f}[v];[1:a]apad[a]",
             "-map", "[v]", "-map", "[a]", "-t", f"{target:.3f}", *VCODEC, "-c:a", "aac", "-b:a", "160k", str(out)])
    else:
        raise SystemExit(f"unknown kind {kind}")
    return out, duration(out), recorded


def srt_time(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries):
    with open(OUT_SRT, "w") as f:
        for i, (t0, t1, text) in enumerate(entries, 1):
            chunks = textwrap.wrap(text, 84) or [""]
            # spread the caption over sub-intervals so lines stay short
            span = (t1 - t0) / max(1, len(chunks)) if chunks else 0
            for j, ch in enumerate(chunks):
                f.write(f"{i}.{j}\n{srt_time(t0 + j*span)} --> {srt_time(t0 + (j+1)*span)}\n{ch}\n\n")


def render_all_cards():
    (BUILD / "cards").mkdir(parents=True, exist_ok=True)
    for seg in SEGMENTS:
        k = seg["kind"]
        fn = {"card": render_card, "receipt": render_receipt, "show": render_show, "show2": render_show2,
              "code": render_code, "image": render_image, "scroll": render_scroll_image}.get(k)
        if fn:
            fn(seg).save(BUILD / "cards" / f"{seg['id']}.png")


def render_slides():
    (BUILD / "slides").mkdir(parents=True, exist_ok=True)
    run(["pdftoppm", "-png", "-scale-to-x", str(W), "-scale-to-y", str(H), str(DECK), str(BUILD / "slides" / "slide")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cards", action="store_true")
    ap.add_argument("--script", action="store_true", help="write recording_script.md: one block per segment to read aloud")
    a = ap.parse_args()
    if a.script:
        with open(HERE / "recording_script.md", "w") as f:
            total = sum(len(g["text"].split()) for g in SEGMENTS if g.get("text")) / TALE_WPS
            f.write("# Recording script\n\nRecord one file per block, saved as audio/<id>.wav. Read the text; "
                    "leave a beat of silence at each end (it is trimmed). Blocks with no text are pauses on a "
                    "receipt and need no file.\n\n"
                    f"Times are a guide at a measured {TALE_WPS} words/sec, the pace this reads best at. They are "
                    "not a budget to hit: a short line with a pause in it ('Eighteen times. She counted.') takes "
                    "far longer than its word count suggests, and should. Read for the sense and ignore the "
                    f"number.\n\nNarration at this pace totals about {total/60:.0f} minutes, plus receipts and "
                    "clips.\n\n")
            for seg in SEGMENTS:
                if seg.get("text"):
                    est = len(seg["text"].split()) / TALE_WPS
                    f.write(f"## {seg['id']}  ({seg['kind']}, about {est:.0f} s at a measured pace)"
                            f"\n\n{seg['text']}\n\n")
        print(HERE / "recording_script.md"); return
    if a.list:
        for seg in SEGMENTS:
            ent = SEG_TAKE.get(seg["id"])
            have = bool(ent and any((TAKES_DIR / ent[0] / f"{ent[1]}{e}").exists()
                                    for e in (".m4a", ".wav", ".mp3", ".mp4"))) \
                or (AUDIO / f"{seg['id']}.wav").exists()
            print(f"{seg['id']:24s} {seg['kind']:8s} {'audio' if have else 'silent':6s} ~{estimate(seg):5.1f}s  {seg.get('text','')[:60]}")
        return
    render_all_cards()
    if a.cards:
        print("cards in", BUILD / "cards"); return
    render_slides()
    seg_dir = BUILD / "seg"; seg_dir.mkdir(parents=True, exist_ok=True)
    entries, t, listing = [], 0.0, []
    built = [build_segment(seg, seg_dir) for seg in SEGMENTS]
    for i, (seg, (out, d, rec)) in enumerate(zip(SEGMENTS, built)):
        # a segment asking for `lead` wants to be on screen before its line:
        # take that time off the END of the previous segment's video, keeping
        # its audio, so the card changes while the last words still play.
        listing.append(f"file '{out}'")
        if seg.get("text"):
            entries.append((t, t + d, seg["text"]))
        print(f"{seg['id']:24s} {d:6.1f}s  {'recorded' if rec else 'silent'}")
        t += d
    (seg_dir / "list.txt").write_text("\n".join(listing) + "\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(seg_dir / "list.txt"), "-c", "copy", str(OUT_MP4)])
    write_srt(entries)
    json.dump([{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()} for s in SEGMENTS],
              open(BUILD / "manifest.json", "w"), indent=1)
    print(f"\n{OUT_MP4}  total {t/60:.1f} min ({t:.0f} s); captions {OUT_SRT.name}")


if __name__ == "__main__":
    main()
