"""Единая схема классов для детектора (круги + слайдеры + спиннеры сразу).

Классы YOLO (фиксированный порядок — менять только с переобучением):
  0 circle        — хит-круг (голова)
  1 slider_head   — голова слайдера
  2 slider_ball   — бегущий шар слайдера (за ним ведем мышь + держим Z/X)
  3 slider_tail   — хвост слайдера
  4 spinner       — спиннер (центр вращения)
  5 cursor        — курсор игрока (для фичей velocity)

Координаты модели: нормализованные xywh относительно infer-кадра 640x360.
urgency для клика: 1 - approach_r / max_approach (считает tracker по кольцу).
"""
from dataclasses import dataclass
from typing import Optional

CLASS_NAMES = ["circle", "slider_head", "slider_ball", "slider_tail", "spinner", "cursor"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

HIT_CLASSES = {"circle", "slider_head"}
HOLD_CLASSES = {"slider_ball"}       # пока ball виден — держим клавишу
END_CLASSES = {"slider_tail"}
SPINNER_CLASSES = {"spinner"}


@dataclass
class Det:
    cls: str
    x: float      # cx 0..1
    y: float      # cy 0..1
    w: float
    h: float
    conf: float
    urgency: Optional[float] = None  # measured approach phase when available
    source: str = "yolo"

    @property
    def cls_id(self) -> int:
        return CLASS_TO_ID.get(self.cls, -1)

    @property
    def is_hittable(self) -> bool:
        return self.cls in HIT_CLASSES

    @property
    def is_hold(self) -> bool:
        return self.cls in HOLD_CLASSES
