"""Mirrors Source/Main/rl_observation.{h,cpp}'s buffer layout (schema v5,
OGRL-20260817-028 Sec4) as named, sliceable regions -- so reward and policy
code never hardcodes a magic float offset that silently goes stale the next
time the C++ side's layout changes. If RLObservation.h's kSchemaVersion is
bumped, this file must be updated to match and SCHEMA_VERSION below bumped
too; ObsLayout.__init__ asserts obs_floats matches what this file computes,
so a mismatch fails loudly at env-connect time instead of misreading floats.

v5 adds the perception fields timing-based play is made of: per-entity
forward(2)/anim_phase(1)/block_health(1)/weapon_type-onehot(5), self
active_blocking(1)/active_block_recharge(1)/weapon_type-onehot(5). Entity
block 24->33, proprioception 28->35, total 260->339. Deliberately does NOT
add an entity.time_in_state field -- see rl_observation.h's kSchemaVersion
comment for why (anim_phase is read identically for entities as for self,
so the "unreliable for non-controlled characters" fallback case the plan
specified it for doesn't apply here).
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 5
LOS_RULE_VERSION = 1

# --- Proprioception (fixed, 35 floats) ---
_PROPRIOCEPTION_FLOATS = 35
KNOCKED_OUT_CLASSES = 3   # awake, unconscious, dead (MovementObject::_awake/_unconscious/_dead)
STATE_CLASSES = 5         # movement, ground, attack, hit_reaction, ragdoll
ACTION_HISTORY_FIELDS = 6  # move_x, move_y, jump, crouch, attack, grab
WEAPON_TYPE_CLASSES = 5   # none, knife, sword, big_sword, spear -- see rl_observation.cpp's WeaponTypeIndex()

# --- Per-entity block (schema v5: 33 floats/entity) ---
ENTITY_FLOATS = (
    1  # valid
    + 1  # id
    + 3  # rel_pos
    + 3  # rel_vel
    + 1  # distance
    + 1  # species
    + KNOCKED_OUT_CLASSES
    + STATE_CLASSES
    + 1  # is_controlled
    + 1  # has_weapon
    + 1  # temp_health         (v2)
    + 1  # blood_health        (v2)
    + 1  # attacked_by_id      (v3) -- MovementObject id of whoever hit this entity most
         # recently (Data/Scripts/aschar.as's attacked_by_id, -1 if nobody has).
         # Exists specifically for reward-causation attribution: compare against
         # self.id before crediting damage/a knockout to THIS agent -- see
         # reward.py, and research-log OGRL-20260816-014 for why this was a
         # real, serious bug without it (the training arena has ~10 characters
         # across multiple teams that fight each other, not just the agent).
    + 1  # is_ally             (v4) -- MovementObject::ASOnSameTeam(self, entity). Closes the
         # same class of bug as attacked_by_id from the other side: causation alone
         # doesn't rule out crediting the agent for hitting its own teammate. See
         # reward.py and research-log OGRL-20260816-015.
    + 2  # fwd.x, fwd.z        (v5) -- egocentric-projected flattened forward vector, so the
         # policy can tell whether this entity is facing it (backstabs, punishes, 1vN positioning).
    + 1  # anim_phase          (v5) -- GetNormalizedAnimTime(); what a parry/punish is timed against.
    + 1  # block_health        (v5) -- a failing guard, pressureable.
    + WEAPON_TYPE_CLASSES  # weapon_type one-hot (v5) -- distinguishes a spear's reach from a knife's.
)


@dataclass(frozen=True)
class ObsLayout:
    max_visible_entities: int = 8
    local_geometry_rays: int = 16
    action_history_steps: int = 4

    def __post_init__(self):
        pass

    @property
    def total_floats(self) -> int:
        return (
            _PROPRIOCEPTION_FLOATS
            + self.action_history_steps * ACTION_HISTORY_FIELDS
            + self.max_visible_entities * ENTITY_FLOATS
            + self.local_geometry_rays
        )

    # --- Proprioception field offsets ---
    SELF_ID = 0
    POS = slice(1, 4)
    VEL = slice(4, 7)
    ANG_VEL = slice(7, 10)
    FORWARD = slice(10, 13)
    GROUNDED = 13
    ANIM_PHASE = 14
    TEMP_HEALTH = 15
    PERMANENT_HEALTH = 16
    BLOOD_HEALTH = 17
    BLOCK_HEALTH = 18
    KNOCKED_OUT = slice(19, 22)  # one-hot: [awake, unconscious, dead]
    STATE = slice(22, 27)        # one-hot: [movement, ground, attack, hit_reaction, ragdoll]
    HAS_WEAPON = 27
    ACTIVE_BLOCKING = 28          # v5
    ACTIVE_BLOCK_RECHARGE = 29    # v5
    WEAPON_TYPE = slice(30, 35)   # v5, one-hot: [none, knife, sword, big_sword, spear]

    @property
    def action_history_start(self) -> int:
        return _PROPRIOCEPTION_FLOATS

    @property
    def entities_start(self) -> int:
        return _PROPRIOCEPTION_FLOATS + self.action_history_steps * ACTION_HISTORY_FIELDS

    @property
    def rays_start(self) -> int:
        return self.entities_start + self.max_visible_entities * ENTITY_FLOATS

    def self_id(self, values: list) -> int:
        return int(values[self.SELF_ID])

    def self_weapon_type_index(self, values: list) -> int:
        """0=none, 1=knife, 2=sword, 3=big_sword, 4=spear -- argmax of the one-hot."""
        onehot = values[self.WEAPON_TYPE]
        return max(range(len(onehot)), key=lambda i: onehot[i])

    def entity_slice(self, slot: int) -> slice:
        start = self.entities_start + slot * ENTITY_FLOATS
        return slice(start, start + ENTITY_FLOATS)

    def entity_field(self, values: list, slot: int) -> dict:
        """Unpacks one entity slot into a named dict. Field order/count must
        match rl_observation.cpp's entity-writing loop exactly."""
        e = values[self.entity_slice(slot)]
        return {
            "valid": bool(e[0]),
            "id": int(e[1]),
            "rel_pos": tuple(e[2:5]),
            "rel_vel": tuple(e[5:8]),
            "distance": e[8],
            "species": int(e[9]),
            "knocked_out": e[10:13],   # one-hot
            "state": e[13:18],         # one-hot
            "is_controlled": bool(e[18]),
            "has_weapon": bool(e[19]),
            "temp_health": e[20],
            "blood_health": e[21],
            "attacked_by_id": int(e[22]),
            "is_ally": bool(e[23]),
            "fwd": tuple(e[24:26]),          # v5: (x, z), egocentric
            "anim_phase": e[26],             # v5
            "block_health": e[27],           # v5
            "weapon_type": e[28:33],         # v5, one-hot
        }

    def entity_weapon_type_index(self, values: list, slot: int) -> int:
        onehot = values[self.entity_slice(slot)][28:33]
        return max(range(len(onehot)), key=lambda i: onehot[i])

    def all_entities(self, values: list) -> list:
        return [self.entity_field(values, slot) for slot in range(self.max_visible_entities)]

    def self_knocked_out_index(self, values: list) -> int:
        """0=awake, 1=unconscious, 2=dead -- argmax of the one-hot."""
        onehot = values[self.KNOCKED_OUT]
        return max(range(len(onehot)), key=lambda i: onehot[i])


DEFAULT_LAYOUT = ObsLayout()
assert DEFAULT_LAYOUT.total_floats == 339, DEFAULT_LAYOUT.total_floats
