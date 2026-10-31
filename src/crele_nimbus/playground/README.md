# The playground's protocol

The page (`static/`) talks to the server over one WebSocket. This protocol is the
playground's own and may change between versions; build on the Python API for anything
else.

**Audio** goes up as binary messages: float32 little-endian PCM at 16 kHz, mono, 1,280
samples (80 ms) per message. The server consumes it 80 ms at a time and answers after
each step.

**Control messages** go up as JSON objects with a `type`:

| Type | Fields | What happens |
|---|---|---|
| `questions` | `questions`: a list of `{id, text, threshold?, start_secs?, stop_threshold?, stop_secs?}`; `signals`: `{user: {settings}, other: {settings}}` | The typed questions, at most 6, each with an id the page chooses (a letter, then letters, digits, `_` or `-`; not `user` or `other`). Criteria go in the text as `question \|\| what yes means \|\| what no means`, or as `yes_means` and `no_means`. A setting left out runs at the default. A question that stays keeps its events; one that goes has its open event stopped, sent with the next `decisions` |
| `enroll` | `pcm`: base64 float32 LE PCM, or `null` | The user's voice for the `user` signal (at least 25 ms, at most 30 s), or forget it |
| `start` | `source: "mic"` | A new stream from the microphone |
| `wav` | `name`, `duration_s` | A new stream from a file the page plays |
| `stop` | `reason?` | The end of the stream |

**The server answers** with JSON:

| Type | Fields |
|---|---|
| `ready` | `state` (`idle` or `listening`), `source`, `name`, `model {name, device}`, `step_ms` (compute per 80 ms at startup, by part), `budget_ms` 80, `defaults`, `questions` (each with `id`, `kind`, `text`, `yes_means`, `no_means`, the four settings it runs with, `cant_tell_below`, and `custom`, the settings set on it), `max_typed`, `enrolled`, `enroll_s`, `sample_rate`, `chunk_samples`. Sent on connect and after every control message that changes one of them |
| `decisions` | per pushed chunk: `steps`, one per 40 ms frame, `{t, t_s, answers: [{id, p, p_can_tell, state}]}` with `state` `yes`, `no` or `cant_tell`; `step_ms`, the stream's mean compute per step; `lag_s`, the audio waiting behind the answers |
| `event` | `event` (`started` or `stopped`), `question`, `text`, `start_s`, `confirmed_s`; for `stopped` also `end_s` and `duration_s` |
| `summary` | on `stop`: `audio_s`, `steps`, `step_ms` by part, `realtime_factor`, `questions` (per question its `events` and time `on_s`), `open` (events still on) |
| `error` | `message`, `where` (the part of the protocol); nothing changed |

One page at a time by default: a second page takes the model over and the first is
closed with code 4000. `--max-sessions` raises that; each page then owns its own session
and they share the model's compute. No audio is stored by the server or by the page.
