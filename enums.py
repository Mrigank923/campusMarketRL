"""Enumerations used by the environment package."""

from enum import Enum


class PhaseEnum(str, Enum):
    MORNING = "morning"
    ACTIVE = "active"
    CLOSING = "closing"


class ShopTypeEnum(str, Enum):
    CAFE = "cafe"
    STATIONARY = "stationary"
    FOOD = "food"
    TECH = "tech"


class TrendTypeEnum(str, Enum):
    NORMAL = "normal"
    FESTIVAL = "festival"
    EXAM = "exam"
    HOLIDAY = "holiday"
