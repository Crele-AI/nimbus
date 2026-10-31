"""Crele Nimbus: real-time yes/no questions about live audio.

```python
from crele_nimbus import Nimbus

model = Nimbus.from_pretrained("crele/nimbus-v1")
session = model.session()
session.ask("music", "Is music playing?")
for update in session.push(pcm_16k):
    for event in update.events:
        print(event.question, event.kind, event.start_s)
```
"""

from crele_nimbus.config import ConfigError, Settings
from crele_nimbus.model import Nimbus
from crele_nimbus.session import Answer, Event, Session, Update
from crele_nimbus.weights import WeightsError

__version__ = "1.0.0"
__all__ = [
    "Answer", "ConfigError", "Event", "Nimbus", "Session", "Settings", "Update",
    "WeightsError", "__version__",
]  # fmt: skip
