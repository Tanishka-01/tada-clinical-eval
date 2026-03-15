from dataclasses import dataclass
from typing import List


@dataclass
class CategoryMeta:
    name: str
    description: str
    count: int
    subcategories: List[str]
    intended_use: str


CATEGORIES = {
    "clinical": CategoryMeta(
        name="clinical",
        description="Medical and clinical speech containing terminology, diagnoses, medications, lab values",
        count=70,
        subcategories=["psychiatry", "cardiology", "endocrinology", "neurology", "oncology", "orthopedics", "general_medicine"],
        intended_use="Clinical voice AI in healthcare settings",
    ),
    "therapeutic": CategoryMeta(
        name="therapeutic",
        description="Warm, empathic, supportive therapeutic speech for mental health applications",
        count=70,
        subcategories=["validation", "grounding", "cognitive_reframing", "encouragement", "psychoeducation", "breathing", "activity"],
        intended_use="CBT-based voice companions and mental health apps",
    ),
    "crisis": CategoryMeta(
        name="crisis",
        description="Careful, calm, de-escalating language for high-stakes emotional moments",
        count=60,
        subcategories=["de-escalation", "resource_provision", "grounding", "presence_validation", "forward_focus"],
        intended_use="Crisis support and de-escalation voice AI",
    ),
}
