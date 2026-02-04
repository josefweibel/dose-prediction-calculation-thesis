"""
Dose-Volume Constraints (DVC) Module

This module handles constraints for organs at risk (OARs) in radiotherapy based on the
CORSAIR guidelines (Bisello et al., 2022). It supports:
- Dose constraints (D_MAX, D_MEAN)
- Volume constraints (V_x where x is dose threshold)
- Dose-volume constraints (D_x where x is volume percentage/absolute)
- Constraint evaluation and metrics calculation

References:
- CORSAIR guidelines: https://doi.org/10.3390/curroncol29100552
"""

from typing import Dict, List, Optional, Literal, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


# Mapping from dataset-specific organ names to constraint library organ names
# This mapping is used for GDPHMM dataset but can be extended for other datasets
GDPHMM_TO_CONSTRAINT_ORGAN = {
  'Cochlea_L': 'Cochlea',
  'Cochlea_R': 'Cochlea',
  'Eyes': 'Eye',
  'Lens_L': 'Lens',
  'Lens_R': 'Lens',
  'OpticNerve_L': 'OpticNerve',
  'OpticNerve_R': 'OpticNerve',
  'Chiasm': 'OpticChiasm',
  'LacrimalGlands': 'LacrimalGland',
  'BrachialPlexus': 'BrachialPlexus',
  'Brain': 'Brain',
  'BrainStem_03': 'Brainstem',
  'Esophagus': 'Esophagus',
  'Lips': 'Lips',
  'Lungs': 'Lung',
  'Trachea': 'Trachea',
  'Larynx-PTV': 'Larynx',
  'Mandible-PTV': 'BoneMandible',
  'ParotidCon-PTV': 'Parotid',
  'ParotidIps-PTV': 'Parotid',
  'Parotids-PTV': 'Parotid',
  'PharConst-PTV': 'PharyngealConstrictorMuscles',
  'Submand-PTV': 'SubmandibularGland',
  'SubmandL-PTV': 'SubmandibularGland',
  'SubmandR-PTV': 'SubmandibularGland',
  'Thyroid-PTV': 'ThyroidGland',
  'SpinalCord_05': 'SpinalCord',
  'OralCavity': 'OralCavity',
  'Submandibular': 'SubmandibularGland',
}


class ConstraintType(Enum):
  """Type of dose-volume constraint."""
  D_MAX = "D_MAX"  # Maximum dose to organ
  D_MEAN = "D_MEAN"  # Mean dose to organ
  V_DOSE = "V_DOSE"  # Volume receiving >= dose threshold
  D_VOLUME = "D_VOLUME"  # Dose to specific volume (percentage or absolute)
  D_CC = "D_CC"  # Dose to specific volume in cubic centimeters


class ConstraintGrade(Enum):
  """Grade/priority of constraint based on clinical evidence."""
  MANDATORY = "mandatory"
  RECOMMENDED = "recommended"
  OPTIMAL = "optimal"
  ACCEPTABLE = "acceptable"


class EvidenceLevel(Enum):
  """Level of evidence for the constraint."""
  A = "A"  # International guidelines
  B = "B"  # Literature reviews on clinical or planning studies
  C = "C"  # Data from results of clinical or planning studies
  D = "D"  # Expert opinions or used in prospective trials


class FractionationType(Enum):
  """Type of radiation fractionation."""
  CONVENTIONAL = "conventional"  # 180-200 cGy/fraction
  HYPO_1 = "hypo_1"  # 1 fraction (SBRT/SRS)
  HYPO_3 = "hypo_3"  # 3 fractions
  HYPO_5 = "hypo_5"  # 5 fractions
  HYPO_8 = "hypo_8"  # 8 fractions


@dataclass
class Constraint:
  """
  Represents a single dose-volume constraint.

  Attributes:
    organ: Name of the organ at risk
    constraint_type: Type of constraint (D_MAX, D_MEAN, V_DOSE, etc.)
    threshold_value: The dose threshold (for V_DOSE) or volume threshold (for D_VOLUME)
    limit_value: The constraint limit value
    comparison: Comparison operator ('<', '<=', '>', '>=')
    grade: Grade/priority of the constraint
    evidence_level: Level of evidence supporting this constraint
    fractionation: Type of fractionation this applies to
    unit: Unit of the threshold/limit ('Gy', '%', 'cm3')
    notes: Additional notes or context
  """
  organ: str
  constraint_type: ConstraintType
  threshold_value: Optional[float] = None  # For V_x or D_x constraints
  limit_value: float = None
  comparison: Literal['<', '<=', '>', '>='] = '<'
  grade: Optional[ConstraintGrade] = None
  evidence_level: Optional[EvidenceLevel] = None
  fractionation: FractionationType = FractionationType.CONVENTIONAL
  unit: str = 'Gy'
  notes: str = ''

  def __str__(self) -> str:
    """Human-readable string representation."""
    if self.constraint_type == ConstraintType.D_MAX:
      return f"{self.organ}: D_MAX {self.comparison} {self.limit_value} {self.unit}"
    elif self.constraint_type == ConstraintType.D_MEAN:
      return f"{self.organ}: D_MEAN {self.comparison} {self.limit_value} {self.unit}"
    elif self.constraint_type == ConstraintType.V_DOSE:
      return f"{self.organ}: V_{self.threshold_value}{self.unit} {self.comparison} {self.limit_value}%"
    elif self.constraint_type in [ConstraintType.D_VOLUME, ConstraintType.D_CC]:
      volume_unit = 'cm3' if self.constraint_type == ConstraintType.D_CC else '%'
      return f"{self.organ}: D_{self.threshold_value}{volume_unit} {self.comparison} {self.limit_value} {self.unit}"
    return f"{self.organ}: {self.constraint_type.value}"

  def to_dict(self) -> dict:
    """Convert to dictionary representation."""
    return {
      'organ': self.organ,
      'constraint_type': self.constraint_type.value,
      'threshold_value': self.threshold_value,
      'limit_value': self.limit_value,
      'comparison': self.comparison,
      'grade': self.grade.value if self.grade else None,
      'evidence_level': self.evidence_level.value if self.evidence_level else None,
      'fractionation': self.fractionation.value,
      'unit': self.unit,
      'notes': self.notes
    }


class ConstraintLibrary:
  """
  Library of standard dose-volume constraints based on CORSAIR guidelines.

  This class stores and manages clinical constraints for various organs at risk.
  Constraints can be filtered by organ, fractionation type, and other criteria.
  """

  def __init__(self):
    self.constraints: List[Constraint] = []
    self._initialize_corsair_constraints()

  def add_constraint(self, constraint: Constraint) -> None:
    """Add a constraint to the library."""
    self.constraints.append(constraint)

  def get_constraints(
    self,
    organ: Optional[str] = None,
    fractionation: Optional[FractionationType] = None,
    constraint_type: Optional[ConstraintType] = None,
    grade: Optional[ConstraintGrade] = None
  ) -> List[Constraint]:
    """
    Get constraints filtered by criteria.

    Args:
      organ: Filter by organ name
      fractionation: Filter by fractionation type
      constraint_type: Filter by constraint type
      grade: Filter by grade/priority

    Returns:
      List of matching constraints
    """
    result = self.constraints

    if organ is not None:
      result = [c for c in result if c.organ.lower() == organ.lower()]

    if fractionation is not None:
      result = [c for c in result if c.fractionation == fractionation]

    if constraint_type is not None:
      result = [c for c in result if c.constraint_type == constraint_type]

    if grade is not None:
      result = [c for c in result if c.grade == grade]

    return result

  def _initialize_corsair_constraints(self) -> None:
    """Initialize constraints based on CORSAIR guidelines (Table 1)."""

    # ============================================================================
    # ANAL CANAL
    # ============================================================================
    self.add_constraint(Constraint(
      organ='AnalCanal',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=40,
      comparison='<',
      evidence_level=EvidenceLevel.C,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='AnalCanal',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=20,
      limit_value=75,
      comparison='<',
      evidence_level=EvidenceLevel.C,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # ANTERIOR DESCENDING ARTERY
    # ============================================================================
    self.add_constraint(Constraint(
      organ='AnteriorDescendingArtery',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=15,
      limit_value=10,
      comparison='<=',
      evidence_level=EvidenceLevel.C,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # BLADDER
    # ============================================================================
    # Conventional fractionation
    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.D_MAX,
      limit_value=65,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For bladder cancer treatment'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=80,
      limit_value=15,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For prostate cancer treatment'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=75,
      limit_value=25,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For prostate cancer treatment'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=70,
      limit_value=35,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For prostate cancer treatment'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=65,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For prostate cancer treatment'
    ))

    # Bladder - Hypofractionated
    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=18.4,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1,
      notes='Dose to bladder-wall'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=28.2,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3,
      notes='Dose to bladder-wall'
    ))

    self.add_constraint(Constraint(
      organ='Bladder',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=38,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5,
      notes='Dose to bladder-wall'
    ))

    # ============================================================================
    # BONE MANDIBLE
    # ============================================================================
    self.add_constraint(Constraint(
      organ='BoneMandible',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=31,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BoneMandible',
      constraint_type=ConstraintType.D_MAX,
      limit_value=70,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BoneMandible',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=55,
      limit_value=20,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # BOWEL
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Bowel',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=15,
      limit_value=120,
      comparison='<',
      unit='cm3',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Bowel',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=45,
      limit_value=195,
      comparison='<',
      unit='cm3',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # BOWEL LARGE
    # ============================================================================
    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=45,
      limit_value=5,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=35,
      limit_value=35,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Bowel Large - Hypofractionated
    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=18.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=28.2,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='BowelLarge',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=38,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # BOWEL SMALL
    # ============================================================================
    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.D_MAX,
      limit_value=55,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=10,
      comparison='<=',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=45,
      limit_value=15,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Bowel Small - Hypofractionated
    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=15.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.5,
      limit_value=25.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.5,
      limit_value=35,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='BowelSmall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.5,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # BRACHIAL PLEXUS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=60,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=66,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Brachial Plexus - Hypofractionated
    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=15,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=17.5,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=24,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=30.5,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='BrachialPlexus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=32,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # BRAIN
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Brain',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=60,
      limit_value=3,
      comparison='<=',
      unit='cm3',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Brain',
      constraint_type=ConstraintType.D_MAX,
      limit_value=72,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Brain - Hypofractionated
    self.add_constraint(Constraint(
      organ='Brain',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=12,
      limit_value=10,
      comparison='<',
      unit='cm3',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1,
      notes='Brain including target'
    ))

    # ============================================================================
    # BRAINSTEM
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_MAX,
      limit_value=54,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Brainstem PRV'
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=1,
      limit_value=59,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='D1-10 cm3 peripheral edge'
    ))

    # Brainstem - Hypofractionated
    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=15,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=10,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=23.1,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=18,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=31,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='Brainstem',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=23,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # CAUDA EQUINA
    # ============================================================================
    self.add_constraint(Constraint(
      organ='CaudaEquina',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=16,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='CaudaEquina',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=24,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='CaudaEquina',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=32,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # CHESTWALL
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Chestwall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.01,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Chestwall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=36.9,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Chestwall',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=43,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # COCHLEA
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=45,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Ideally one side'
    ))

    # Cochlea - Hypofractionated
    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=9,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=4,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MAX,
      limit_value=12,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=17.1,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MAX,
      limit_value=20,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=25,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='Cochlea',
      constraint_type=ConstraintType.D_MAX,
      limit_value=27.5,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # DUODENUM
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.D_MAX,
      limit_value=55,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=10,
      comparison='<=',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Duodenum - Hypofractionated
    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=12.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=22.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=35,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='Duodenum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=33,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # ESOPHAGUS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=34,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=35,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=40,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=70,
      limit_value=20,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Esophagus - Hypofractionated
    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=15.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=25.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Esophagus',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=35,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # EYE
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Eye',
      constraint_type=ConstraintType.D_MAX,
      limit_value=45,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Retina/Macula'
    ))

    # Eye - Hypofractionated
    self.add_constraint(Constraint(
      organ='Eye',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=8,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1,
      notes='Macula and Retina'
    ))

    # ============================================================================
    # EYE ANTERIOR (LENS, CORNEA)
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Lens',
      constraint_type=ConstraintType.D_MAX,
      limit_value=4,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Cornea',
      constraint_type=ConstraintType.D_MAX,
      limit_value=40,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Lens - Hypofractionated
    self.add_constraint(Constraint(
      organ='Lens',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=1.5,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    # ============================================================================
    # FEMORAL HEAD-NECK
    # ============================================================================
    self.add_constraint(Constraint(
      organ='FemoralHeadNeck',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=44,
      limit_value=5,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='FemoralHeadNeck',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=40,
      limit_value=35,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='FemoralHeadNeck',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # GLOTTIS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Glottis',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=50,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Glottis',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=27,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Glottis',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=44,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Glottis',
      constraint_type=ConstraintType.D_MAX,
      limit_value=73.5,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # GREAT VESSELS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='GreatVessels',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='GreatVessels',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=45,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='GreatVessels',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=53,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # HEART
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=26,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=25,
      limit_value=10,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=30,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Heart - Hypofractionated
    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.03,
      limit_value=22,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.5,
      limit_value=26,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Heart',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.5,
      limit_value=24,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    # ============================================================================
    # KIDNEYS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Kidneys',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=18,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Kidneys',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=20,
      limit_value=33,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Kidneys - Hypofractionated
    self.add_constraint(Constraint(
      organ='Kidneys',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=10,
      limit_value=33,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Kidneys',
      constraint_type=ConstraintType.D_CC,
      threshold_value=200,
      limit_value=16,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Kidneys',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=10,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # LACRIMAL GLAND
    # ============================================================================
    self.add_constraint(Constraint(
      organ='LacrimalGland',
      constraint_type=ConstraintType.D_MAX,
      limit_value=40,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='LacrimalGland',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=26,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # LARYNX
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Larynx',
      constraint_type=ConstraintType.D_MAX,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Larynx',
      constraint_type=ConstraintType.D_MAX,
      limit_value=66,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Larynx',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Larynx',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=27,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # LIVER
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=25,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=33,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Liver - Hypofractionated
    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_CC,
      threshold_value=700,
      limit_value=9.1,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=15,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=13,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_CC,
      threshold_value=700,
      limit_value=17,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Liver',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=15.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # LUNG
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=40,
      limit_value=10,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=15,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=20,
      limit_value=20,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=10,
      limit_value=40,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=5,
      limit_value=50,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=20,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Lung - Hypofractionated
    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=20,
      limit_value=15,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1,
      notes='Lungs and Lungs-ITV'
    ))

    self.add_constraint(Constraint(
      organ='Lung',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=8,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    # ============================================================================
    # OPTIC CHIASM
    # ============================================================================
    self.add_constraint(Constraint(
      organ='OpticChiasm',
      constraint_type=ConstraintType.D_MAX,
      limit_value=55,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # OPTIC NERVE
    # ============================================================================
    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_MAX,
      limit_value=55,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Optic Nerve - Hypofractionated
    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=10,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=8,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=20,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=15,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=25,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='OpticNerve',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=22.5,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # ORAL CAVITY
    # ============================================================================
    self.add_constraint(Constraint(
      organ='OralCavity',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=73,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Limit D_MEAN to uninvolved oral cavity'
    ))

    # ============================================================================
    # PAROTID GLAND
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Parotid',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=25,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For both glands'
    ))

    self.add_constraint(Constraint(
      organ='Parotid',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=20,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='For single gland'
    ))

    self.add_constraint(Constraint(
      organ='Parotid',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=30,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Parotid',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=40,
      limit_value=33,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Contralateral'
    ))

    # ============================================================================
    # PHARYNGEAL CONSTRICTOR MUSCLES
    # ============================================================================
    self.add_constraint(Constraint(
      organ='PharyngealConstrictorMuscles',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # RECTUM
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=60,
      limit_value=35,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=65,
      limit_value=25,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=70,
      limit_value=20,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=75,
      limit_value=15,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Rectum - Hypofractionated
    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.01,
      limit_value=18.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=28.2,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Rectum',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=38,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # SKIN
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Skin',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.03,
      limit_value=25,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL,
      notes='To avoid local permanent alopecia'
    ))

    # Skin - Hypofractionated
    self.add_constraint(Constraint(
      organ='Skin',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.01,
      limit_value=26,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Skin',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=33,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Skin',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=39.5,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # SPINAL CANAL / SPINAL CORD
    # ============================================================================
    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_MAX,
      limit_value=45,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_MAX,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Spinal Canal/Cord - Hypofractionated
    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=14,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=12.4,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=20.3,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=18,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=25.3,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_MAX,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.035,
      limit_value=32,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_8
    ))

    self.add_constraint(Constraint(
      organ='SpinalCanal',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=25,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_8
    ))

    # SpinalCord uses same constraints as SpinalCanal
    for constraint in [c for c in self.constraints if c.organ == 'SpinalCanal']:
      self.add_constraint(Constraint(
        organ='SpinalCord',
        constraint_type=constraint.constraint_type,
        threshold_value=constraint.threshold_value,
        limit_value=constraint.limit_value,
        comparison=constraint.comparison,
        grade=constraint.grade,
        evidence_level=constraint.evidence_level,
        fractionation=constraint.fractionation,
        unit=constraint.unit,
        notes=constraint.notes
      ))

    # ============================================================================
    # STOMACH
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_MAX,
      limit_value=54,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=45,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=50,
      limit_value=10,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=45,
      limit_value=15,
      comparison='<=',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # Stomach - Hypofractionated
    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=12.4,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=22.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=35,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='Stomach',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=33,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # SUBMANDIBULAR GLAND
    # ============================================================================
    self.add_constraint(Constraint(
      organ='SubmandibularGland',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=35,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    # ============================================================================
    # THYROID GLAND
    # ============================================================================
    self.add_constraint(Constraint(
      organ='ThyroidGland',
      constraint_type=ConstraintType.V_DOSE,
      threshold_value=45,
      limit_value=50,
      comparison='<',
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='ThyroidGland',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=45,
      comparison='<',
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL,
      notes='Or sparing at least 5 cm3 of the thyroid < 45 Gy'
    ))

    # ============================================================================
    # TRACHEA
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Trachea',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=20.2,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_1
    ))

    self.add_constraint(Constraint(
      organ='Trachea',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_3
    ))

    self.add_constraint(Constraint(
      organ='Trachea',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=38,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    self.add_constraint(Constraint(
      organ='Trachea',
      constraint_type=ConstraintType.D_CC,
      threshold_value=0.1,
      limit_value=35,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.A,
      fractionation=FractionationType.HYPO_5
    ))

    # ============================================================================
    # LIPS
    # ============================================================================
    self.add_constraint(Constraint(
      organ='Lips',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=30,
      comparison='<',
      grade=ConstraintGrade.OPTIMAL,
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL
    ))

    self.add_constraint(Constraint(
      organ='Lips',
      constraint_type=ConstraintType.D_MEAN,
      limit_value=50,
      comparison='<',
      grade=ConstraintGrade.MANDATORY,
      evidence_level=EvidenceLevel.B,
      fractionation=FractionationType.CONVENTIONAL
    ))


class ConstraintEvaluator:
  """
  Evaluates dose distributions against clinical constraints.

  This class computes dose-volume metrics and checks compliance with constraints.
  """

  def __init__(self, constraints: Optional[List[Constraint]] = None):
    """
    Initialize evaluator with constraints.

    Args:
      constraints: List of constraints to evaluate. If None, uses default library.
    """
    if constraints is None:
      library = ConstraintLibrary()
      self.constraints = library.constraints
    else:
      self.constraints = constraints

  def compute_d_max(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute maximum dose in the region.

    Args:
      dose: Dose distribution array
      mask: Optional binary mask defining the region

    Returns:
      Maximum dose value in Gy
    """
    if mask is not None:
      masked_dose = dose[mask > 0]
      if masked_dose.size == 0:
        return 0.0
      return float(np.max(masked_dose))
    return float(np.max(dose))

  def compute_d_mean(self, dose: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute mean dose in the region.

    Args:
      dose: Dose distribution array
      mask: Optional binary mask defining the region

    Returns:
      Mean dose value in Gy
    """
    if mask is not None:
      masked_dose = dose[mask > 0]
      if masked_dose.size == 0:
        return 0.0
      return float(np.mean(masked_dose))
    return float(np.mean(dose))

  def compute_v_dose(
    self,
    dose: np.ndarray,
    dose_threshold: float,
    mask: Optional[np.ndarray] = None
  ) -> float:
    """
    Compute volume percentage receiving at least the dose threshold.

    Args:
      dose: Dose distribution array
      dose_threshold: Dose threshold in Gy
      mask: Optional binary mask defining the region

    Returns:
      Volume percentage (0-100)
    """
    if mask is not None:
      masked_dose = dose[mask > 0]
      if masked_dose.size == 0:
        return 0.0
      return float(100.0 * np.sum(masked_dose >= dose_threshold) / masked_dose.size)
    else:
      total_voxels = dose.size
      if total_voxels == 0:
        return 0.0
      return float(100.0 * np.sum(dose >= dose_threshold) / total_voxels)

  def compute_d_volume(
    self,
    dose: np.ndarray,
    volume_percent: float,
    mask: Optional[np.ndarray] = None
  ) -> float:
    """
    Compute dose to a specific volume percentage.

    Args:
      dose: Dose distribution array
      volume_percent: Volume percentage (0-100)
      mask: Optional binary mask defining the region

    Returns:
      Dose value in Gy received by that volume percentage
    """
    if mask is not None:
      masked_dose = dose[mask > 0]
    else:
      masked_dose = dose.flatten()

    if masked_dose.size == 0:
      return 0.0

    # Sort doses in descending order
    sorted_doses = np.sort(masked_dose)[::-1]

    # Find index corresponding to volume percentage
    idx = int(np.ceil(volume_percent / 100.0 * len(sorted_doses))) - 1
    idx = max(0, min(idx, len(sorted_doses) - 1))

    return float(sorted_doses[idx])

  def compute_d_cc(
    self,
    dose: np.ndarray,
    volume_cc: float,
    mask: Optional[np.ndarray] = None,
    voxel_volume_cc: float = 1.0
  ) -> float:
    """
    Compute dose to a specific absolute volume in cubic centimeters.

    Args:
      dose: Dose distribution array
      volume_cc: Volume in cubic centimeters
      mask: Optional binary mask defining the region
      voxel_volume_cc: Volume of a single voxel in cm3

    Returns:
      Dose value in Gy received by that volume
    """
    if mask is not None:
      masked_dose = dose[mask > 0]
    else:
      masked_dose = dose.flatten()

    if masked_dose.size == 0:
      return 0.0

    # Convert volume from cc to number of voxels
    n_voxels = int(np.ceil(volume_cc / voxel_volume_cc))

    # Sort doses in descending order
    sorted_doses = np.sort(masked_dose)[::-1]

    # Get dose at the specified volume
    idx = min(n_voxels - 1, len(sorted_doses) - 1)

    return float(sorted_doses[idx])

  def evaluate_constraint(
    self,
    constraint: Constraint,
    dose: np.ndarray,
    mask: Optional[np.ndarray] = None,
    voxel_volume_cc: float = 1.0
  ) -> Dict[str, Union[float, bool, str]]:
    """
    Evaluate a single constraint against a dose distribution.

    Args:
      constraint: Constraint to evaluate
      dose: Dose distribution array
      mask: Optional binary mask for the organ
      voxel_volume_cc: Volume of a single voxel in cm3

    Returns:
      Dictionary with evaluation results:
        - 'value': computed metric value
        - 'limit': constraint limit value
        - 'satisfied': whether constraint is satisfied
        - 'margin': how much margin to the limit (positive = safe)
        - 'description': human-readable description
    """
    # Compute the metric based on constraint type
    if constraint.constraint_type == ConstraintType.D_MAX:
      value = self.compute_d_max(dose, mask)
    elif constraint.constraint_type == ConstraintType.D_MEAN:
      value = self.compute_d_mean(dose, mask)
    elif constraint.constraint_type == ConstraintType.V_DOSE:
      value = self.compute_v_dose(dose, constraint.threshold_value, mask)
    elif constraint.constraint_type == ConstraintType.D_VOLUME:
      value = self.compute_d_volume(dose, constraint.threshold_value, mask)
    elif constraint.constraint_type == ConstraintType.D_CC:
      value = self.compute_d_cc(dose, constraint.threshold_value, mask, voxel_volume_cc)
    else:
      raise ValueError(f"Unknown constraint type: {constraint.constraint_type}")

    # Check if constraint is satisfied
    limit = constraint.limit_value
    comparison = constraint.comparison

    if comparison == '<':
      satisfied = value < limit
      margin = limit - value
    elif comparison == '<=':
      satisfied = value <= limit
      margin = limit - value
    elif comparison == '>':
      satisfied = value > limit
      margin = value - limit
    elif comparison == '>=':
      satisfied = value >= limit
      margin = value - limit
    else:
      raise ValueError(f"Unknown comparison operator: {comparison}")

    return {
      'value': value,
      'limit': limit,
      'satisfied': satisfied,
      'margin': margin,
      'description': str(constraint),
      'constraint': constraint
    }

  def evaluate_all_constraints(
    self,
    dose: np.ndarray,
    organ_masks: Dict[str, np.ndarray],
    voxel_volume_cc: float = 1.0,
    fractionation: FractionationType = FractionationType.CONVENTIONAL
  ) -> Dict[str, List[Dict]]:
    """
    Evaluate all relevant constraints for multiple organs.

    Args:
      dose: Dose distribution array
      organ_masks: Dictionary mapping organ names to binary masks
      voxel_volume_cc: Volume of a single voxel in cm3
      fractionation: Type of fractionation used

    Returns:
      Dictionary mapping organ names to lists of evaluation results
    """
    results = {}

    for organ_name, mask in organ_masks.items():
      # Get constraints for this organ and fractionation
      organ_constraints = [
        c for c in self.constraints
        if c.organ.lower() == organ_name.lower() and c.fractionation == fractionation
      ]

      if not organ_constraints:
        continue

      organ_results = []
      for constraint in organ_constraints:
        result = self.evaluate_constraint(constraint, dose, mask, voxel_volume_cc)
        organ_results.append(result)

      results[organ_name] = organ_results

    return results

  def compute_constraint_violation_score(
    self,
    dose: np.ndarray,
    organ_masks: Dict[str, np.ndarray],
    voxel_volume_cc: float = 1.0,
    fractionation: FractionationType = FractionationType.CONVENTIONAL
  ) -> float:
    """
    Compute overall constraint violation score.

    This metric quantifies how much the dose distribution violates constraints.
    Lower is better, with 0 meaning all constraints are satisfied.

    Args:
      dose: Dose distribution array
      organ_masks: Dictionary mapping organ names to binary masks
      voxel_volume_cc: Volume of a single voxel in cm3
      fractionation: Type of fractionation used

    Returns:
      Violation score (0 = all constraints satisfied)
    """
    results = self.evaluate_all_constraints(dose, organ_masks, voxel_volume_cc, fractionation)

    total_violation = 0.0
    n_constraints = 0

    for organ_results in results.values():
      for result in organ_results:
        n_constraints += 1
        if not result['satisfied']:
          # Normalize violation by the limit value to make it scale-independent
          if result['limit'] != 0:
            violation = abs(result['margin']) / result['limit']
          else:
            violation = abs(result['margin'])
          total_violation += violation

    if n_constraints == 0:
      return 0.0

    return total_violation / n_constraints

  def get_summary_report(
    self,
    dose: np.ndarray,
    organ_masks: Dict[str, np.ndarray],
    voxel_volume_cc: float = 1.0,
    fractionation: FractionationType = FractionationType.CONVENTIONAL
  ) -> str:
    """
    Generate a human-readable summary report of constraint evaluations.

    Args:
      dose: Dose distribution array
      organ_masks: Dictionary mapping organ names to binary masks
      voxel_volume_cc: Volume of a single voxel in cm3
      fractionation: Type of fractionation used

    Returns:
      Formatted summary report as string
    """
    results = self.evaluate_all_constraints(dose, organ_masks, voxel_volume_cc, fractionation)

    lines = [
      "=" * 80,
      f"Dose-Volume Constraint Evaluation Report",
      f"Fractionation: {fractionation.value}",
      "=" * 80,
      ""
    ]

    for organ_name, organ_results in results.items():
      if not organ_results:
        continue

      lines.append(f"\n{organ_name}:")
      lines.append("-" * 80)

      for result in organ_results:
        status = "✓ PASS" if result['satisfied'] else "✗ FAIL"
        constraint = result['constraint']

        # Format based on grade
        grade_str = f" [{constraint.grade.value}]" if constraint.grade else ""
        evidence_str = f" (Level {constraint.evidence_level.value})" if constraint.evidence_level else ""

        lines.append(
          f"  {status} {result['description']}{grade_str}{evidence_str}"
        )
        lines.append(
          f"     Actual value: {result['value']:.2f}, "
          f"Limit: {result['limit']:.2f}, "
          f"Margin: {result['margin']:+.2f}"
        )

    # Add summary statistics
    all_results = [r for organ_results in results.values() for r in organ_results]
    n_total = len(all_results)
    n_passed = sum(1 for r in all_results if r['satisfied'])
    violation_score = self.compute_constraint_violation_score(
      dose, organ_masks, voxel_volume_cc, fractionation
    )

    lines.extend([
      "",
      "=" * 80,
      f"Summary: {n_passed}/{n_total} constraints satisfied",
      f"Overall violation score: {violation_score:.4f}",
      "=" * 80
    ])

    return "\n".join(lines)
