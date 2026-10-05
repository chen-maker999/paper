"""ALFWorld adapter: state writes from a text-game interaction.

ALFWorld (Shridhar et al., 2021) is a household text game: the agent searches
receptacles, picks objects up, transforms them (clean / heat / cool / light)
and places them. The working state an agent must keep is small but keeps being
rewritten: what each receptacle contains and whether it is open, what the agent
holds, and which objects have been transformed. Forgetting it makes the agent
re-search places it has already visited; keeping a superseded copy (an item
that has since been taken) sends it back to an empty receptacle.

`AlfWorldState` is the domain extractor: it consumes (action, observation)
pairs and emits one `Write` per state key the step touched. Keys are
receptacles (``recep:drawer 1``), the inventory (``inv``) and transformed
objects (``obj:pan 1``). Values are rendered from the extractor's current
belief (the last observation of a receptacle with later take/move deltas
applied), so a ledger record is always the latest value of its key.
"""
from __future__ import annotations

import re

from .core import Write

_NAME = r"[a-z][a-z ]*? \d+"
_ITEMS = re.compile(r"\ban? ([a-z][a-z]*? \d+)")
_ARRIVE = re.compile(rf"^You arrive at ({_NAME})\.")
_ON = re.compile(rf"On the ({_NAME}), you see (.*?)\.(?:\s|$)")
_IN = re.compile(r"In it, you see (.*?)\.(?:\s|$)")
_STATE = re.compile(rf"The ({_NAME}) is (open|closed)\.")
_OPEN = re.compile(rf"^You open the ({_NAME})\.")
_CLOSE = re.compile(rf"^You close the ({_NAME})\.")
_TAKE = re.compile(rf"^You pick up the ({_NAME}) from the ({_NAME})\.")
_MOVE = re.compile(rf"^You (?:move|put) the ({_NAME}) (?:to|in|on|in/on) the ({_NAME})\.")
_XFORM = re.compile(rf"^You (clean|heat|cool) the ({_NAME}) using the ({_NAME})\.")
_USE = re.compile(rf"^You turn (on|off) the ({_NAME})\.")
_CARRY = re.compile(r"^You are carrying: (.*?)\.?$")
_EMPTY_HANDS = "You are not carrying anything."
_FACING = re.compile(rf"^You are facing the ({_NAME})\.")

_PAST = {"clean": "cleaned", "heat": "heated", "cool": "cooled"}


def items(text: str) -> list[str]:
    return [] if text.strip() == "nothing" else _ITEMS.findall(text)


def receptacles(initial_obs: str) -> list[str]:
    m = re.search(r"you see (.*?)\.\s*(?:\n|$)", initial_obs)
    return items(m.group(1)) if m else []


class AlfWorldState:
    """Belief state of the agent, updated from observations; emits writes."""

    def __init__(self):
        self.contents: dict[str, list[str] | None] = {}
        self.openness: dict[str, str] = {}
        self.holding: list[str] = []
        self.status: dict[str, list[str]] = {}
        self.where = None

    # ---------------------------------------------------------------- render
    def recep_value(self, r: str) -> str:
        st = self.openness.get(r)
        c = self.contents.get(r)
        if c is None:
            return f"{r} is {st}, contents not seen" if st else f"{r}: contents not seen"
        body = ", ".join(c) if c else "nothing"
        return f"{r}{' (' + st + ')' if st else ''} contains: {body}"

    def inv_value(self) -> str:
        if not self.holding:
            return "you are carrying nothing"
        return "you are carrying: " + ", ".join(self._obj(o) for o in self.holding)

    def _obj(self, o: str) -> str:
        return f"{o} ({', '.join(self.status[o])})" if self.status.get(o) else o

    def obj_value(self, o: str) -> str:
        return f"{o} has been {', '.join(self.status[o])}"

    # ---------------------------------------------------------------- update
    def _see(self, r: str, listing: str):
        self.contents[r] = items(listing)

    def update(self, action: str, obs: str, turn: int) -> list[Write]:
        touched_r, touched_inv, touched_o = [], False, []
        o = obs.strip()
        m = _ARRIVE.match(o)
        if m:
            self.where = m.group(1)
        m = _FACING.match(o)
        if m:
            self.where = m.group(1)
        for m in _ON.finditer(o):
            self._see(m.group(1), m.group(2)); touched_r.append(m.group(1))
        for m in _STATE.finditer(o):
            r = m.group(1)
            self.openness[r] = m.group(2)
            if m.group(2) == "open":
                mi = _IN.search(o, m.end())
                if mi:
                    self._see(r, mi.group(1))
            touched_r.append(r)
        m = _OPEN.match(o)
        if m:
            self.openness[m.group(1)] = "open"; touched_r.append(m.group(1))
        m = _CLOSE.match(o)
        if m:
            self.openness[m.group(1)] = "closed"; touched_r.append(m.group(1))
        m = _TAKE.match(o)
        if m:
            obj, r = m.groups()
            if self.contents.get(r) and obj in self.contents[r]:
                self.contents[r].remove(obj)
            self.holding.append(obj)
            touched_r.append(r); touched_inv = True
        m = _MOVE.match(o)
        if m:
            obj, r = m.groups()
            if obj in self.holding:
                self.holding.remove(obj)
            if self.contents.get(r) is not None:
                self.contents[r].append(obj)
            else:
                self.contents[r] = [obj]
            touched_r.append(r); touched_inv = True
        m = _XFORM.match(o)
        if m:
            verb, obj, _ = m.groups()
            st = self.status.setdefault(obj, [])
            if _PAST[verb] not in st:
                st.append(_PAST[verb])
            touched_o.append(obj); touched_inv = True
        m = _USE.match(o)
        if m:
            self.status[m.group(2)] = ["turned " + m.group(1)]
            touched_o.append(m.group(2))
        m = _CARRY.match(o)
        if m:
            self.holding = items(m.group(1)); touched_inv = True
        if o == _EMPTY_HANDS:
            self.holding = []; touched_inv = True
        writes = []
        for r in dict.fromkeys(touched_r):
            writes.append(Write(f"recep:{r}", "recep", self.recep_value(r), turn, (self.recep_value(r),)))
        for ob in dict.fromkeys(touched_o):
            writes.append(Write(f"obj:{ob}", "obj", self.obj_value(ob), turn, (self.obj_value(ob),)))
        if touched_inv:
            writes.append(Write("inv", "inv", self.inv_value(), turn, (self.inv_value(),)))
        return writes
