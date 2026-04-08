#!/usr/bin/env python3
from __future__ import annotations

"""
Convert predicted dose .npy files to DICOM RTDose format.

Multiple models (including the original dose) are exported into a single
DICOM directory per patient, with shared CT / RTSTRUCT / RTPLAN and one
RTDose file per model.

Output structure:
  {output_dir}/
    {patient_id}/
      CT/         -- CT slice DICOMs (shared)
      RTSTRUCT/   -- structure-set DICOM (shared)
      RTPLAN/     -- plan DICOM (shared)
      RTDOSE/
        RD.{model_name}.dcm   (one per model)

Usage:
  python scripts/convert_dose_to_dicom.py \
    --model variant_l:samples/165-ws-rfm-combi-fp32/flow-cache-beamdiff-enc2-interval5-20-200-inf-2-bf16-amp/test_199 \
    --model variant_t:original \
    --model variant_v:samples/101-beamdiff/ddpm-1000/test_199 \
    --patient HNSCC-01-0090   # optional: process only this patient
"""

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DICOM discovery helpers
# ---------------------------------------------------------------------------

def find_patient_dicom_dir(dicom_root: Path, patient_id: str) -> Path | None:
    """
    Look up the patient directory directly under dicom_root.
    Falls back to a shallow scan if the exact name doesn't exist.
    """
    direct = dicom_root / patient_id
    if direct.is_dir():
        return direct
    # Fallback: check immediate children
    for p in dicom_root.iterdir():
        if p.is_dir() and patient_id in p.name:
            return p
    return None


def find_dicom_files_by_modality(patient_dir: Path) -> dict[str, list[Path]]:
    """
    Use the modality subdirectory names (CT/, RTDOSE/, RTPLAN/, RTSTRUCT/)
    to group .dcm files — no DICOM header reads needed.
    """
    result: dict[str, list[Path]] = {}
    for subdir in patient_dir.iterdir():
        if not subdir.is_dir():
            continue
        modality = subdir.name  # CT, RTDOSE, RTPLAN, RTSTRUCT
        dcm_files = sorted(subdir.glob('*.dcm'))
        if dcm_files:
            result[modality] = dcm_files
    return result



# ---------------------------------------------------------------------------
# Dose array helpers
# ---------------------------------------------------------------------------

def reverse_padding(
    dose_zyx_padded: np.ndarray,
    orig_z: int,
    orig_y: int,
    pad_size: tuple[int, int],
) -> np.ndarray:
    """
    Reverse the symmetric zero-padding applied by preprocess_gdphmm_slices.py.
    Padding was applied to axes 0 (z) and 1 (y) only; axis 2 (x) is untouched.
    dose_zyx_padded: (Z_pad, Y_pad, X)
    Returns:         (orig_z, orig_y, X)
    """
    pz = pad_size[0] - orig_z
    py = pad_size[1] - orig_y
    z_off = pz // 2
    y_off = py // 2
    return dose_zyx_padded[z_off:z_off + orig_z, y_off:y_off + orig_y, :]


def write_rtdose_dicom(
    template_ds: pydicom.Dataset,
    dose_zyx: np.ndarray,
    image_position: tuple[float, float, float],
    voxel_spacing: tuple[float, float, float],
    output_path: Path,
    model_name: str = '',
) -> None:
    """
    Create an RTDOSE DICOM from template_ds with new dose data and coordinates.

    template_ds:    original RTDOSE dataset (used for patient demographics, UIDs, etc.)
    dose_zyx:       (Z, Y, X) dose in Gy — already in DICOM frame order
    image_position: (x, y, z) mm — ImagePositionPatient (from vol['origin'])
    voxel_spacing:  (x_mm, y_mm, z_mm) — from vol['spacing']
    """
    ds = copy.deepcopy(template_ds)
    Z, Y, X = dose_zyx.shape
    x_sp, y_sp, z_sp = voxel_spacing

    # Array is already in DICOM frame order (Z, Rows=Y, Cols=X) — no transpose
    dose = dose_zyx.astype(np.float64)
    dose = np.clip(dose, 0, None)  # ensure non-negative

    max_dose = float(dose.max())
    scale = max_dose / (2**32 - 1) if max_dose > 0 else 1.0
    pixels = (dose / scale).astype(np.uint32)

    # Pixel data
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.DoseGridScaling = scale
    ds.NumberOfFrames = Z
    ds.Rows = Y
    ds.Columns = X
    ds.PixelData = pixels.tobytes()

    # Spatial metadata — PixelSpacing is [row_spacing, col_spacing] = [y_sp, x_sp]
    ds.ImagePositionPatient = list(image_position)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [y_sp, x_sp]
    ds.GridFrameOffsetVector = [i * z_sp for i in range(Z)]

    # Identify the model/variant in viewer-visible tags
    if model_name:
        ds.SeriesDescription = model_name
        ds.DoseComment = model_name

    # New SOP Instance UID + Series UID (each model is its own series)
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))


# ---------------------------------------------------------------------------
# File-copy helper
# ---------------------------------------------------------------------------

def copy_dicom_files(src_files: list[Path], dst_dir: Path) -> None:
    """Copy DICOM files into dst_dir, preserving filenames."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        shutil.copy2(str(src), str(dst_dir / src.name))


# ---------------------------------------------------------------------------
# GDPHMM .npz helper
# ---------------------------------------------------------------------------

def load_gdphmm_vol(gdphmm_dir: Path, sample_name: str) -> dict | None:
    """
    Load a GDPHMM .npz volume dict.  Tries exact filename first, then falls
    back to any file in the directory that starts with the patient ID.
    """
    patient_id = sample_name.split('+')[0]
    exact = gdphmm_dir / f'{sample_name}.npz'
    if exact.exists():
        return np.load(str(exact), allow_pickle=True)['arr_0'].item()
    matches = sorted(gdphmm_dir.glob(f'{patient_id}+*.npz'))
    if not matches:
        return None
    return np.load(str(matches[0]), allow_pickle=True)['arr_0'].item()


# ---------------------------------------------------------------------------
# Collect patient IDs from all model sources
# ---------------------------------------------------------------------------

def collect_patient_samples(
    models: list[tuple[str, str]],
    gdphmm_dir: Path,
    file_glob: str,
) -> dict[str, dict[str, str]]:
    """
    Build a mapping: patient_id → {model_name: sample_name_or_'original'}.
    models: list of (model_name, samples_path_or_'original')
    Returns only patients that appear in at least one model source.
    """
    patients: dict[str, dict[str, str]] = {}

    for model_name, source in models:
        if source == 'original':
            for npz_path in sorted(gdphmm_dir.glob('*.npz')):
                sample_name = npz_path.stem
                pid = sample_name.split('+')[0]
                patients.setdefault(pid, {})[model_name] = sample_name
        else:
            samples_dir = Path(source)
            for npy_path in sorted(samples_dir.glob(file_glob)):
                sample_name = npy_path.stem
                pid = sample_name.split('+')[0]
                patients.setdefault(pid, {})[model_name] = sample_name

    return patients


# ---------------------------------------------------------------------------
# Per-patient processing (all models at once)
# ---------------------------------------------------------------------------

def process_patient(
    patient_id: str,
    model_doses: dict[str, str],
    models: list[tuple[str, str]],
    gdphmm_dir: Path,
    dicom_root: Path,
    output_dir: Path,
    pad_size: tuple[int, int],
    copy_plan: bool = False,
) -> bool:
    """
    Process one patient: write all model RTDoses + shared CT/RTSTRUCT/RTPLAN.
    model_doses: {model_name: sample_name}
    models: [(model_name, source_path_or_'original'), ...]
    """
    prefix = f'  [{patient_id}]'

    # Find patient DICOM directory
    patient_dir = find_patient_dicom_dir(dicom_root, patient_id)
    if patient_dir is None:
        print(f'{prefix} no DICOM directory found — skipping')
        return False
    modality_files = find_dicom_files_by_modality(patient_dir)

    # Get the original RTDOSE — used as template
    if 'RTDOSE' not in modality_files:
        print(f'{prefix} no original RTDOSE found — skipping')
        return False
    template_ds = pydicom.dcmread(str(modality_files['RTDOSE'][0]))

    # Optionally find the RTPLAN matching the plan code from the sample filename
    relevant_plans: list[Path] = []
    if copy_plan:
        if 'RTPLAN' not in modality_files:
            print(f'{prefix} no RTPLAN found — skipping plan copy')
        else:
            any_sample = next(iter(model_doses.values()))
            parts = any_sample.split('+')
            plan_code = parts[1] if len(parts) > 1 else None

            if plan_code:
                for plan_path in modality_files['RTPLAN']:
                    ds = pydicom.dcmread(str(plan_path), stop_before_pixels=True)
                    label = str(getattr(ds, 'RTPlanLabel', ''))
                    if label.startswith(plan_code):
                        relevant_plans.append(plan_path)
            if not relevant_plans:
                relevant_plans = modality_files['RTPLAN']  # fallback: copy all

    out_dir = output_dir / patient_id

    # Build a lookup: model_name → source path
    source_map = {name: source for name, source in models}

    # Write one RTDose per model
    dose_written = 0
    for model_name, sample_name in model_doses.items():
        source = source_map[model_name]

        # Load GDPHMM vol (needed for origin + original shape)
        vol = load_gdphmm_vol(gdphmm_dir, sample_name)
        if vol is None:
            print(f'{prefix} .npz not found for {model_name} — skipping dose')
            continue

        if source == 'original':
            # Original dose from .npz
            dose_gy = (vol['dose'] * vol['dose_scale'] * vol['Body']).astype(np.float32)
        else:
            # Predicted dose from .npy — already in Gy
            samples_dir = Path(source)
            npy_candidates = sorted(samples_dir.glob(f'{sample_name}.*'))
            npy_path = None
            for c in npy_candidates:
                if c.suffix == '.npy' and not c.stem.endswith('_sigma'):
                    npy_path = c
                    break
            if npy_path is None:
                npy_path = samples_dir / f'{sample_name}.npy'
            if not npy_path.exists():
                print(f'{prefix} {npy_path} not found — skipping {model_name}')
                continue

            dose_raw = np.load(str(npy_path))
            if dose_raw.ndim == 4:
                dose_raw = dose_raw[0]  # drop channel dim → (Z_pad, Y_pad, X)

            # Reverse padding (axes 0=z and 1=y were padded; axis 2=x untouched)
            orig_z, orig_y = vol['img'].shape[:2]
            dose_gy = reverse_padding(dose_raw, orig_z, orig_y, pad_size)

        # Use origin and spacing directly from NPZ — both in (x, y, z) order
        image_position = tuple(float(v) for v in vol['origin'])
        voxel_spacing = tuple(float(v) for v in vol['spacing'])

        write_rtdose_dicom(
            template_ds, dose_gy, image_position, voxel_spacing,
            out_dir / 'RTDOSE' / f'RD.{model_name}.dcm',
            model_name=model_name,
        )
        dose_written += 1

    if dose_written == 0:
        print(f'{prefix} no doses written — skipping shared files')
        return False

    # Copy shared CT, RTSTRUCT (and optionally RTPLAN)
    for modality, subdir in [('CT', 'CT'), ('RTSTRUCT', 'RTSTRUCT')]:
        if modality in modality_files:
            copy_dicom_files(modality_files[modality], out_dir / subdir)
    if relevant_plans:
        copy_dicom_files(relevant_plans, out_dir / 'RTPLAN')

    ct_count = len(modality_files.get('CT', []))
    print(f'{prefix} done  {dose_written} dose(s)  CT={ct_count}')
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert dose predictions to DICOM RTDose (multiple models per patient).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model', action='append', required=True, metavar='NAME:PATH',
        help='Model specification as name:path (use "original" as path for GDPHMM ground truth). '
             'Can be repeated for multiple models.',
    )
    parser.add_argument(
        '--dicom-root', type=Path, default=Path('/storage/research/artorg_mia/Head_and_Neck'),
        help='Root of the TCIA DICOM data',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('dicom/'),
        help='Output root directory',
    )
    parser.add_argument(
        '--gdphmm-dir', type=Path, default=Path('data/gdphmm-hn-vmat/test'),
        help='GDPHMM split directory containing the original .npz volume files',
    )
    parser.add_argument(
        '--pad-size', type=int, nargs=2, default=[176, 160], metavar=('H', 'W'),
        help='H×W padding target used during preprocessing (default: 176 160)',
    )
    parser.add_argument(
        '--file-glob', default='*.npy',
        help='Glob pattern for selecting prediction files (default: *.npy)',
    )
    parser.add_argument(
        '--patient', type=str, default=None,
        help='Process only this patient ID',
    )
    parser.add_argument(
        '--copy-plan', action='store_true', default=False,
        help='Copy matching RTPLAN files into output (skipped by default)',
    )
    args = parser.parse_args()

    # Parse --model arguments
    models: list[tuple[str, str]] = []
    for spec in args.model:
        if ':' not in spec:
            parser.error(f'Invalid --model format "{spec}" — expected NAME:PATH')
        name, source = spec.split(':', 1)
        models.append((name, source))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pad_size = tuple(args.pad_size)

    # Collect all patients across all model sources
    patients = collect_patient_samples(models, args.gdphmm_dir, args.file_glob)

    # Filter to single patient if requested; otherwise skip already-converted
    if args.patient:
        if args.patient not in patients:
            raise SystemExit(f'Patient {args.patient} not found in any model source')
        patients = {args.patient: patients[args.patient]}
    else:
        model_names = {name for name, _ in models}
        skipped = 0
        filtered: dict[str, dict[str, str]] = {}
        for pid, model_doses in patients.items():
            rtdose_dir = args.output_dir / pid / 'RTDOSE'
            existing = {p.stem.removeprefix('RD.') for p in rtdose_dir.glob('RD.*.dcm')} if rtdose_dir.exists() else set()
            if model_names <= existing:
                skipped += 1
            else:
                filtered[pid] = model_doses
        if skipped:
            print(f'Skipping {skipped} already-converted patient(s)')
        patients = filtered

    print(f'{len(patients)} patient(s), {len(models)} model(s)')
    for name, source in models:
        print(f'  {name}: {source}')

    successes = sum(
        process_patient(
            pid, model_doses, models,
            args.gdphmm_dir, args.dicom_root, args.output_dir,
            pad_size, copy_plan=args.copy_plan,
        )
        for pid, model_doses in tqdm(sorted(patients.items()), desc='Patients', leave=False)
    )
    print(f'\nDone: {successes}/{len(patients)} patients exported.')


if __name__ == '__main__':
    main()
