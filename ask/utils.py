import json, re
from typing import Optional

def extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    raw = m.group()
    for candidate in (raw, raw.replace("'", '"')):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None
