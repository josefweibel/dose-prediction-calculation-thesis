#!/usr/bin/env python3
"""
Script to scan multiple cohort directories under ~/artorg_mia/Head_and_Neck/
and extract detailed RTPLAN metadata to a single CSV file.
"""

import pydicom
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


def extract_patient_id_from_path(file_path: Path) -> str:
  """Extract patient ID from file path."""
  parts = file_path.parts
  for part in reversed(parts):
    if any(pattern in part.upper() for pattern in ['HNSCC', 'LUNG', 'HN-', 'HNC_', '0522', '0617']):
      return part
  if len(parts) >= 2:
    return parts[-2]
  return "unknown"


def determine_treatment_modality(beams: List[Dict]) -> Dict[str, str]:
  """Determine treatment modality from beam configuration."""
  if not beams:
    return {'modality': 'UNKNOWN', 'technique': 'UNKNOWN'}

  beam_types = set()
  has_vmat = False
  total_control_points = 0
  gantry_rotations = []
  all_gantry_angles = []
  rotation_directions = set()

  for beam in beams:
    beam_type = beam.get('beam_type', '')
    beam_types.add(beam_type)

    # Check for gantry rotation direction (CW/CC indicates VMAT)
    rotation_dir = beam.get('gantry_rotation_direction', '')
    if rotation_dir:
      rotation_directions.add(rotation_dir)
    if rotation_dir in ['CW', 'CC']:
      has_vmat = True

    if beam.get('control_points'):
      control_points = beam['control_points']
      total_control_points += len(control_points)

      gantry_angles = [cp.get('gantry_angle') for cp in control_points
               if cp.get('gantry_angle') is not None]
      all_gantry_angles.extend(gantry_angles)

      if len(gantry_angles) > 1:
        angle_changes = [abs(gantry_angles[i+1] - gantry_angles[i])
                 for i in range(len(gantry_angles)-1)]
        if any(change > 5 for change in angle_changes):
          gantry_rotations.append(max(gantry_angles) - min(gantry_angles))

    if 'DYNAMIC' in beam_type.upper():
      has_vmat = True

  # Determine modality based on multiple factors
  if has_vmat or (gantry_rotations and total_control_points > 50):
    modality = 'VMAT'
    technique = 'Volumetric Modulated Arc Therapy'
  elif total_control_points > 10:
    modality = 'IMRT'
    technique = 'Intensity Modulated Radiation Therapy'
  elif total_control_points <= 10:
    modality = '3D-CRT'
    technique = '3D Conformal Radiation Therapy'
  else:
    modality = 'UNKNOWN'
    technique = 'Unknown Technique'

  result = {
    'modality': modality,
    'technique': technique,
    'total_control_points': total_control_points,
  }

  if gantry_rotations:
    result['arc_rotation_degrees'] = gantry_rotations

  if all_gantry_angles:
    result['gantry_angles'] = all_gantry_angles

  if rotation_directions:
    result['rotation_directions'] = rotation_directions

  return result


def extract_rtplan_detailed_info(rtplan_path: str) -> Dict:
  """Extract detailed information from RTPLAN DICOM file."""
  try:
    ds = pydicom.dcmread(rtplan_path)
    file_path = Path(rtplan_path)

    # Basic identifiers
    patient_id = str(ds.PatientID) if hasattr(ds, 'PatientID') else extract_patient_id_from_path(file_path)

    # Determine plan_id: prefer RTPlanLabel, fall back to RTPlanName, then SOPInstanceUID
    plan_id = None
    if hasattr(ds, 'RTPlanLabel') and ds.RTPlanLabel:
      plan_id = str(ds.RTPlanLabel)
    elif hasattr(ds, 'RTPlanName') and ds.RTPlanName:
      plan_id = str(ds.RTPlanName)
    elif hasattr(ds, 'SOPInstanceUID'):
      plan_id = str(ds.SOPInstanceUID)

    info = {
      'patient_id': patient_id,
      'plan_id': plan_id,
      'plan_name': str(ds.RTPlanName) if hasattr(ds, 'RTPlanName') else None,
      'plan_label': str(ds.RTPlanLabel) if hasattr(ds, 'RTPlanLabel') else None,
      'file_name': file_path.name,
      'file_path': str(file_path),
      'cohort': file_path.parts[-4] if len(file_path.parts) >= 4 else None,  # e.g., TCIA4_HNSCC
    }

    # Fractionation
    if hasattr(ds, 'FractionGroupSequence'):
      total_fractions = sum(
        int(fg.NumberOfFractionsPlanned)
        for fg in ds.FractionGroupSequence
        if hasattr(fg, 'NumberOfFractionsPlanned')
      )
      info['total_fractions'] = total_fractions
      info['fraction_groups'] = len(ds.FractionGroupSequence)
    else:
      info['total_fractions'] = None
      info['fraction_groups'] = None

    # Prescription dose
    if hasattr(ds, 'DoseReferenceSequence'):
      prescriptions = []
      for dr in ds.DoseReferenceSequence:
        if hasattr(dr, 'TargetPrescriptionDose'):
          prescriptions.append(float(dr.TargetPrescriptionDose))

      info['prescription_dose_gy'] = prescriptions[0] if prescriptions else None
      info['num_prescription_targets'] = len(prescriptions)

      if prescriptions and info.get('total_fractions'):
        info['dose_per_fraction_gy'] = prescriptions[0] / info['total_fractions']
    else:
      info['prescription_dose_gy'] = None
      info['num_prescription_targets'] = None
      info['dose_per_fraction_gy'] = None

    # Beam information
    if hasattr(ds, 'BeamSequence'):
      energies = set()
      machine_names = set()
      beam_types = set()
      dose_rates = []
      mlc_info = []
      beams_for_modality = []

      isocenters = []

      for beam in ds.BeamSequence:
        # Machine info
        if hasattr(beam, 'TreatmentMachineName'):
          machine_names.add(str(beam.TreatmentMachineName))
        if hasattr(beam, 'Manufacturer'):
          if 'manufacturer' not in info:
            info['manufacturer'] = str(beam.Manufacturer)

        beam_type = str(beam.BeamType) if hasattr(beam, 'BeamType') else None
        if beam_type:
          beam_types.add(beam_type)

        # Beam for modality determination
        beam_dict = {
          'beam_type': beam_type,
          'control_points': []
        }

        # Control points
        if hasattr(beam, 'ControlPointSequence'):
          for i, cp in enumerate(beam.ControlPointSequence):
            # Energy
            if hasattr(cp, 'NominalBeamEnergy'):
              energies.add(float(cp.NominalBeamEnergy))

            # Dose rate
            if hasattr(cp, 'DoseRateSet'):
              dose_rates.append(float(cp.DoseRateSet))

            # Gantry rotation direction (for VMAT detection)
            gantry_rotation_dir = None
            if hasattr(cp, 'GantryRotationDirection'):
              gantry_rotation_dir = str(cp.GantryRotationDirection)
              if i == 0:  # Store from first control point
                beam_dict['gantry_rotation_direction'] = gantry_rotation_dir

            # Isocenter position (from first control point)
            if i == 0 and hasattr(cp, 'IsocenterPosition'):
              isocenter = [float(x) for x in cp.IsocenterPosition]
              isocenters.append(isocenter)

            # Gantry angle for modality
            cp_dict = {
              'gantry_angle': float(cp.GantryAngle) if hasattr(cp, 'GantryAngle') else None,
            }
            beam_dict['control_points'].append(cp_dict)

        beams_for_modality.append(beam_dict)

        # MLC info
        if hasattr(beam, 'BeamLimitingDeviceSequence'):
          for bld in beam.BeamLimitingDeviceSequence:
            if hasattr(bld, 'RTBeamLimitingDeviceType'):
              device_type = str(bld.RTBeamLimitingDeviceType)
              if device_type == 'MLCX' and hasattr(bld, 'NumberOfLeafJawPairs'):
                mlc_info.append(int(bld.NumberOfLeafJawPairs))

      # Store beam information
      info['number_of_beams'] = len(ds.BeamSequence)
      info['beam_energies_mv'] = ','.join(map(str, sorted(energies))) if energies else None
      info['num_unique_energies'] = len(energies)
      info['machine_name'] = ','.join(machine_names) if machine_names else None
      info['beam_types'] = ','.join(beam_types) if beam_types else None

      if dose_rates:
        info['dose_rate_min'] = min(dose_rates)
        info['dose_rate_max'] = max(dose_rates)
      else:
        info['dose_rate_min'] = None
        info['dose_rate_max'] = None

      if mlc_info:
        info['mlc_leaves'] = mlc_info[0]  # Take first
      else:
        info['mlc_leaves'] = None

      # Store isocenter information
      if isocenters:
        # Store first isocenter (most plans use same isocenter for all beams)
        info['isocenter_x'] = isocenters[0][0]
        info['isocenter_y'] = isocenters[0][1]
        info['isocenter_z'] = isocenters[0][2]
        info['num_isocenters'] = len(set([tuple(iso) for iso in isocenters]))
      else:
        info['isocenter_x'] = None
        info['isocenter_y'] = None
        info['isocenter_z'] = None
        info['num_isocenters'] = None

      # Determine modality
      modality_info = determine_treatment_modality(beams_for_modality)
      info['modality'] = modality_info['modality']
      info['technique'] = modality_info['technique']
      info['total_control_points'] = modality_info['total_control_points']

      if 'arc_rotation_degrees' in modality_info:
        arc_rotations = modality_info['arc_rotation_degrees']
        info['max_arc_rotation'] = max(arc_rotations) if arc_rotations else None
      else:
        info['max_arc_rotation'] = None

      if 'gantry_angles' in modality_info:
        gantry_angles = modality_info['gantry_angles']
        if gantry_angles:
          info['gantry_angles_list'] = ','.join(map(str, sorted(set(gantry_angles))))
          info['num_unique_gantry_angles'] = len(set(gantry_angles))
        else:
          info['gantry_angles_list'] = None
          info['num_unique_gantry_angles'] = None
      else:
        info['gantry_angles_list'] = None
        info['num_unique_gantry_angles'] = None

      if 'rotation_directions' in modality_info:
        rot_dirs = modality_info['rotation_directions']
        info['rotation_directions'] = ','.join(sorted(rot_dirs)) if rot_dirs else None
      else:
        info['rotation_directions'] = None
    else:
      info['number_of_beams'] = None
      info['modality'] = None
      info['technique'] = None

    # DICOM UIDs (for tracking)
    info['study_instance_uid'] = str(ds.StudyInstanceUID) if hasattr(ds, 'StudyInstanceUID') else None
    info['sop_instance_uid'] = str(ds.SOPInstanceUID) if hasattr(ds, 'SOPInstanceUID') else None

    return info

  except Exception as e:
    return {
      'patient_id': extract_patient_id_from_path(Path(rtplan_path)),
      'file_path': str(rtplan_path),
      'error': str(e)
    }


def scan_head_neck_cohorts(base_dir: str, output_csv: str = "head_neck_rtplan_metadata.csv"):
  """
  Scan all cohort directories in ~/artorg_mia/Head_and_Neck/ (except GDPHMM).
  Extract detailed RTPLAN metadata and save to CSV.

  Args:
    base_dir: Base directory (e.g., ~/artorg_mia/Head_and_Neck/)
    output_csv: Output CSV file path
  """
  print("="*80)
  print("HEAD-AND-NECK COHORTS RTPLAN METADATA EXTRACTION")
  print("="*80)

  base_path = Path(base_dir).expanduser()

  if not base_path.exists():
    print(f"❌ Directory not found: {base_path}")
    return

  print(f"\nScanning: {base_path}")

  # Find all subdirectories (cohorts) except GDPHMM
  cohort_dirs = [d for d in base_path.iterdir()
           if d.is_dir() and d.name.upper() != 'GDPHMM']

  print(f"Found {len(cohort_dirs)} cohort directories:")
  for cohort in cohort_dirs:
    print(f"  - {cohort.name}")

  # Find all RTPLAN files across all cohorts
  print(f"\nSearching for RTPLAN files...")
  all_rtplan_files = []

  for cohort_dir in tqdm(cohort_dirs, desc="Scanning cohorts"):
    # Find RTPLAN directories
    for rtplan_dir in cohort_dir.rglob("RTPLAN"):
      if rtplan_dir.is_dir():
        rtplan_files = list(rtplan_dir.glob("*.dcm"))
        all_rtplan_files.extend(rtplan_files)

    # Also find RP.*.dcm files
    rp_files = list(cohort_dir.rglob("RP.*.dcm"))
    all_rtplan_files.extend(rp_files)

  # Remove duplicates
  all_rtplan_files = list(set(all_rtplan_files))

  print(f"Found {len(all_rtplan_files)} RTPLAN files")

  if len(all_rtplan_files) == 0:
    print("❌ No RTPLAN files found!")
    return

  # Extract information from each file
  print(f"\nExtracting metadata from RTPLAN files...")

  metadata_list = []
  errors = []

  for rtplan_file in tqdm(all_rtplan_files, desc="Processing RTPLAN files"):
    try:
      info = extract_rtplan_detailed_info(str(rtplan_file))
      metadata_list.append(info)

      if 'error' in info:
        errors.append((rtplan_file.name, info['error']))
    except Exception as e:
      errors.append((rtplan_file.name, str(e)))
      tqdm.write(f"    ⚠️  Error processing {rtplan_file.name}: {e}")

  # Create DataFrame
  df = pd.DataFrame(metadata_list)

  # Save to CSV
  df.to_csv(output_csv, index=False)

  print(f"\n{'='*80}")
  print("EXTRACTION COMPLETE")
  print(f"{'='*80}")
  print(f"\nOutput file: {output_csv}")
  print(f"Total entries: {len(df)}")

  if errors:
    print(f"Errors encountered: {len(errors)}")

  # Print statistics
  print(f"\n{'='*80}")
  print("DATASET STATISTICS")
  print(f"{'='*80}")

  # Cohort breakdown
  if 'cohort' in df.columns:
    print("\nPlans by Cohort:")
    cohort_counts = df['cohort'].value_counts()
    for cohort, count in cohort_counts.items():
      print(f"  {cohort}: {count}")

  # Patient count
  if 'patient_id' in df.columns:
    print(f"\nUnique patients: {df['patient_id'].nunique()}")

  # Modality breakdown
  if 'modality' in df.columns:
    print("\nTreatment Modalities:")
    modality_counts = df['modality'].value_counts()
    for modality, count in modality_counts.items():
      percentage = count / len(df) * 100
      print(f"  {modality}: {count} ({percentage:.1f}%)")

  # Fractionation
  if 'total_fractions' in df.columns:
    valid_fractions = df['total_fractions'].dropna()
    if len(valid_fractions) > 0:
      print(f"\nFractionation:")
      print(f"  Mean: {valid_fractions.mean():.1f}")
      print(f"  Median: {valid_fractions.median():.1f}")
      print(f"  Range: {valid_fractions.min():.0f} - {valid_fractions.max():.0f}")

  # Beam energies
  if 'beam_energies_mv' in df.columns:
    all_energies = set()
    for energies_str in df['beam_energies_mv'].dropna():
      all_energies.update(energies_str.split(','))
    if all_energies:
      print(f"\nBeam Energies: {', '.join(sorted(all_energies))} MV")

  # Machines
  if 'machine_name' in df.columns:
    all_machines = set()
    for machines_str in df['machine_name'].dropna():
      all_machines.update(machines_str.split(','))
    if all_machines:
      print(f"\nTreatment Machines:")
      for machine in sorted(all_machines):
        print(f"  - {machine}")

  print(f"\n{'='*80}")
  print(f"✅ Metadata extraction complete!")
  print(f"CSV file saved to: {output_csv}")
  print(f"{'='*80}\n")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Extract detailed RTPLAN metadata from Head-and-Neck cohorts"
  )
  parser.add_argument(
    '--base-dir',
    type=str,
    default='~/artorg_mia/Head_and_Neck',
    help='Base directory containing cohort subdirectories (default: ~/artorg_mia/Head_and_Neck)'
  )
  parser.add_argument(
    '--output',
    type=str,
    default='data/gdphmm-hn/incoming/dicom_metadata.csv',
    help='Output CSV file (default: dicom_metadata.csv)'
  )

  args = parser.parse_args()

  # Run extraction
  scan_head_neck_cohorts(args.base_dir, args.output)
