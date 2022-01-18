import dataclasses

from typing import List

@dataclasses.dataclass
class SeedCamSpace:
    x_c: float #X center
    y_c: float #Y Center
    h  : float #Height
    w  : float #Width
    p  : float #Probality

SeedList = List[SeedCamSpace]

@dataclasses.dataclass
class SeedUTMSpace:
    n_c  : float
    e_c  : float
    zone : int
    zone_letter : str
    h    : float
    w    : float
    p    : float


@dataclasses.dataclass
class SeedDistStruct:
    seed     : SeedUTMSpace
    count    :    int
    distance : float
    frame_idx : int