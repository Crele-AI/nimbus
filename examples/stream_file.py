#!/usr/bin/env python3
"""Stream an audio file through Crele Nimbus and print the events as they happen.

    python examples/stream_file.py recording.wav -q "Is music playing?" -q "Is someone laughing?"
    python examples/stream_file.py call.wav --voice me.wav      # makes the "user" row answer

The file must be 16 kHz mono (resample first, for example with
`ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav`). Audio is pushed 80 ms at a time, as a live
stream would arrive; `--realtime` paces it at the speed of the audio.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import soundfile as sf

from crele_nimbus import Nimbus

CHUNK = 1280  # 80 ms at 16 kHz


def read_16k_mono(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000:
        sys.exit(f"{path} is {rate} Hz; resample it to 16 kHz mono first")
    return audio.mean(axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", help="a 16 kHz mono audio file")
    ap.add_argument(
        "-q", "--question", action="append", default=[],
        help="a yes/no question (repeat for more); default: music and laughter",
    )  # fmt: skip
    ap.add_argument("--voice", help="a clip of the user's voice, for the 'user' signal")
    ap.add_argument("--weights", default="crele/nimbus-v1", help="a repository id or a folder")
    ap.add_argument("--realtime", action="store_true", help="pace the audio at real time")
    args = ap.parse_args()
    questions = args.question or ["Is music playing?", "Is someone laughing?"]

    model = Nimbus.from_pretrained(args.weights)
    session = model.session()
    for i, text in enumerate(questions, 1):
        session.ask(f"q{i}", text)
    if args.voice:
        session.enroll(read_16k_mono(args.voice))
    names = dict(session.questions)

    audio = read_16k_mono(args.audio)
    print(f"{args.audio}: {len(audio) / 16000:.1f} s, {len(questions)} questions")
    started = time.perf_counter()
    for i in range(0, len(audio), CHUNK):
        for update in session.push(audio[i : i + CHUNK]):
            for e in update.events:
                when = f"{e.start_s:6.2f} s"
                if e.kind == "started":
                    print(f"{when}  YES  {names[e.question]}  (confirmed at {e.confirmed_s:.2f} s)")
                else:
                    print(f"{when}  no   {names[e.question]}  (after {e.duration_s:.2f} s)")
        if args.realtime:
            time.sleep(max(0.0, started + (i + CHUNK) / 16000 - time.perf_counter()))
    for update in session.finish():
        for e in update.events:
            word = "YES" if e.kind == "started" else "no "
            print(f"{e.start_s:6.2f} s  {word}  {names[e.question]}")
    for e in session.events:
        if e.kind == "started":
            print(f"{e.start_s:6.2f} s  YES  {names[e.question]}  (still on at the end)")
    t = session.timing
    print(f"compute: {t['total']:.1f} ms per 80 ms of audio")


if __name__ == "__main__":
    main()
