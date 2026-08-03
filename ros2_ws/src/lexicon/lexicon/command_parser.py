"""Turn a natural-language instruction into a target phrase for the detector.

Open-vocabulary detectors (OWL-ViT, Grounding DINO, YOLO-World) take a free-text
query, so the parser's job is small: strip the command verb and filler, keep the
object description including attributes ("red", "small"), and return one or more
candidate phrases ranked by specificity.

This is deliberately a rules-based parser with no model dependency so it runs
instantly on a robot and is trivial to test. For messier phrasing you can swap
in an LLM: set an env var and route through llm_extract() (see the stub). The
open-vocab detector does the heavy lifting either way, so the parser does not
need to be clever.
"""

from __future__ import annotations

import re
from typing import List

# Leading command patterns we strip. Order matters: longer phrases first.
_COMMAND_PREFIXES = [
    r"^please\s+",
    r"^can you\s+",
    r"^go\s+(?:to|towards|toward)\s+",
    r"^navigate\s+(?:to|towards|toward)\s+",
    r"^drive\s+(?:to|towards|toward)\s+",
    r"^move\s+(?:to|towards|toward)\s+",
    r"^head\s+(?:to|towards|toward)\s+",
    r"^bring me\s+(?:to\s+)?",
    r"^take me\s+(?:to\s+)?",
    r"^find\s+(?:me\s+)?",
    r"^look for\s+",
    r"^search for\s+",
    r"^locate\s+",
    r"^approach\s+",
    r"^where(?:'s| is)\s+",
    r"^get\s+(?:to\s+)?",
]

# "nearest"/"closest" are selectors (see wants_nearest), not visual attributes,
# so they are stripped from the phrase sent to the detector. Matches a lone
# article at end of string too, so "go to the" reduces to empty.
_LEADING_ARTICLES = re.compile(r"^(the|a|an|my|that|this|nearest|closest|farthest|furthest)(\s+|$)", re.I)
_TRAILING_JUNK = re.compile(r"[.?!]+$")


def parse_instruction(text: str) -> List[str]:
    """Return candidate target phrases, most specific first.

    "go to the red chair"       -> ["red chair", "chair"]
    "find me the nearest mug"   -> ["mug"]
    "where is the backpack?"    -> ["backpack"]
    """
    s = text.strip().lower()
    s = _TRAILING_JUNK.sub("", s)

    # Strip stacked command prefixes ("please find ...", "can you go to ...").
    changed = True
    while changed:
        changed = False
        for pat in _COMMAND_PREFIXES:
            new = re.sub(pat, "", s, flags=re.I)
            if new != s:
                s = new.strip()
                changed = True
                break

    # Strip leading articles and selectors, repeatedly ("the nearest mug").
    changed = True
    while changed:
        new = _LEADING_ARTICLES.sub("", s).strip()
        changed = new != s
        s = new

    if not s:
        return []

    phrases = [s]

    # Add a backoff phrase without the leading adjective, so if "red chair"
    # finds nothing we can still try "chair". Only add if there is an adjective.
    tokens = s.split()
    if len(tokens) >= 2:
        head_noun = tokens[-1]
        if head_noun != s:
            phrases.append(head_noun)

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def wants_nearest(text: str) -> bool:
    """True if the user asked for the nearest/closest instance, which changes how
    we pick among multiple detections (closest by depth rather than most confident)."""
    return bool(re.search(r"\b(nearest|closest)\b", text, re.I))


def wants_farthest(text: str) -> bool:
    """True if the user asked for the farthest/furthest instance, which picks
    among multiple detections by largest depth rather than most confident."""
    return bool(re.search(r"\b(farthest|furthest)\b", text, re.I))


def llm_extract(text: str) -> List[str]:
    """Optional LLM-backed parser for messy instructions. Stub by design.

    Wire this to your LLM of choice and return a list of target phrases. Keeping
    it behind a function boundary means the node code does not change when you
    upgrade from rules to an LLM.
    """
    raise NotImplementedError("plug in an LLM here; parse_instruction covers the common cases")
