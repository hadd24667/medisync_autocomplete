from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PatientContext:
    age: Optional[int] = None           # tuổi (năm)
    sex: Optional[str] = None           # "M" / "F" / None
    specialty: Optional[str] = None     # "pediatrics", "cardiology", "nutrition", ...
    active_icd: List[str] = field(default_factory=list)   # list ICD text/keyword
    active_atc: List[str] = field(default_factory=list)   # list mã ATC đang dùng
    allergies: List[str] = field(default_factory=list)    # text: "penicillin", "aspirin", ...
