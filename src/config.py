from dataclasses import dataclass
import math

@dataclass
class Config:
    # ---------- World ----------
    DT: float = 0.1
    WORLD_MIN: float = 0.0
    WORLD_MAX: float = 1.0
    STEPS_PER_EPISODE: int = 1000

    # ---------- Population ----------
    POP_SIZE: int = 50
    ELITE_FRAC: float = 0.05

    # ---------- Agent energy ----------
    START_ENERGY: float = 3.0
    BASE_ENERGY_LOSS: float = 0.006
    SPEED_ENERGY_COST: float = 0.03
    TURN_COST: float = 0.002


    # ---------- Movement ----------
    TURN_SMOOTH: float = 0.8
    TURN_RESPONSE: float = 0.2
    MAX_SPEED_SCALE: float = 0.08
    

    # ---------- Vision / sensing ----------
    VISION_RADIUS: float = 0.25
    VISION_ANGLE: float = math.pi / 2
    FOOD_SIGNAL_GAIN: float = 5.0

    # ---------- Food ----------
    FOOD_CLUMPS: int = 5
    FOOD_PER_CLUMP: int = 10
    FOOD_SPREAD: float = 0.03
    FOOD_ENERGY: float = 1.0
    EAT_RADIUS: float = 0.01
    FOOD_RESPAWN_RATE: float = 0.007


    # ---------- RNN ----------
    INPUTS: int = 8
    HIDDEN: int = 12
    OUTPUTS: int = 2
    W_INIT_SCALE: float = 0.4
    WHH_INIT_SCALE: float = 0.2

    # ---------- Evolution ----------
    MUT_SIGMA: float = 0.01
    RECURRENT_MUT_SCALE: float = 0.1
