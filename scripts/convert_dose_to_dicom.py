#!/usr/bin/env python3
"""
Convert predicted dose .npy files to DICOM RTDose format.

Two modes:

  Predicted dose:
    Reads combined .npy volume files, reverses the preprocessing padding,
    resamples the dose to the original RTDOSE coordinate grid, and writes a
    new RTDOSE DICOM.  CT, RTSTRUCT, and RTPLAN are copied alongside as-is.

  Original dose (--original):
    Copies the original RTDOSE DICOM (no resampling) together with CT,
    RTSTRUCT, and RTPLAN.  Both modes produce the same output structure so
    predicted and original dose share the same coordinate grid.

Output structure:
  {output_dir}/
    {model_name}/
      {patient_id}/
        CT/         -- CT slice DICOMs
        RTSTRUCT/   -- structure-set DICOM
        RTPLAN/     -- plan DICOM
        RTDOSE/     -- RD.{patient_id}.dcm  (predicted or original)

Usage:
  # Export a model's predictions
  python convert_dose_to_dicom.py \\
    --samples-dir samples/MyModel/sampling/test_100 \\
    --model-name MyModel \\
    --dicom-root /storage/.../Head_and_Neck \\
    --gdphmm-dir data/gdphmm-hn/test \\
    --output-dir dicom_output

  # Export the original dose (same structure)
  python convert_dose_to_dicom.py \\
    --original \\
    --model-name original \\
    --dicom-root /storage/.../Head_and_Neck \\
    --gdphmm-dir data/gdphmm-hn/test \\
    --output-dir dicom_output
"""

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pydicom
import scipy.ndimage
from tqdm import tqdm

# Dose normalisation range from src/data.py
GDPHMM_DOSE_MAX = 88.0  # Gy


# ---------------------------------------------------------------------------
# DICOM discovery helpers
# ---------------------------------------------------------------------------

def find_patient_dicom_dir(dicom_root: Path, patient_id: str) -> Path | None:
    """
    Walk the TCIA tree and return the shallowest directory whose name
    contains patient_id.
    """
    candidates = [
        p for p in dicom_root.rglob('*')
        if p.is_dir() and patient_id in p.name
    ]
    if not candidates:
        return None
    # Prefer the shallowest (closest to root) match
    return min(candidates, key=lambda p: len(p.parts))


def find_dicom_files_by_modality(patient_dir: Path) -> dict[str, list[Path]]:
    """
    Recursively scan patient_dir for .dcm files and group them by Modality tag.
    Returns e.g. {'CT': [...], 'RTDOSE': [...], 'RTSTRUCT': [...], 'RTPLAN': [...]}.
    Files that cannot be read are silently skipped.
    """
    result: dict[str, list[Path]] = {}
    for dcm_path in sorted(patient_dir.rglob('*.dcm')):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            modality = str(ds.Modality)
            result.setdefault(modality, []).append(dcm_path)
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# RTDOSE grid helpers
# ---------------------------------------------------------------------------

def get_rtdose_grid(
    ds: pydicom.Dataset,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """
    Return (shape, spacing) from an RTDOSE dataset.
    shape   = (D, H, W) = (NumberOfFrames, Rows, Columns)
    spacing = (row_mm, col_mm, z_mm)
    """
    n_frames = int(ds.NumberOfFrames)
    rows = int(ds.Rows)
    cols = int(ds.Columns)
    row_sp = float(ds.PixelSpacing[0])
    col_sp = float(ds.PixelSpacing[1])
    offsets = [float(x) for x in ds.GridFrameOffsetVector]
    if len(offsets) > 1:
        z_sp = abs(offsets[1] - offsets[0])
    else:
        z_sp = float(ds.get('SliceThickness', 1.0))
    return (n_frames, rows, cols), (row_sp, col_sp, z_sp)


# ---------------------------------------------------------------------------
# Dose array helpers
# ---------------------------------------------------------------------------

def reverse_padding(
    dose_hwz_padded: np.ndarray,
    orig_h: int,
    orig_w: int,
    pad_size: tuple[int, int],
) -> np.ndarray:
    """
    Reverse the symmetric zero-padding applied by preprocess_gdphmm_slices.py.
    dose_hwz_padded: (H_pad, W_pad, D)
    Returns:         (orig_h, orig_w, D)
    """
    ph = pad_size[0] - orig_h
    pw = pad_size[1] - orig_w
    top = ph // 2
    left = pw // 2
    return dose_hwz_padded[top:top + orig_h, left:left + orig_w, :]


def resample_dose(
    dose_hwz: np.ndarray,
    src_spacing: tuple[float, float, float],
    tgt_spacing: tuple[float, float, float],
    tgt_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Resample a dose volume from GDPHMM space to the original RTDOSE grid.

    dose_hwz:    (H, W, D) array in Gy, GDPHMM voxel space
    src_spacing: (row_mm, col_mm, z_mm) for GDPHMM axes H, W, D
    tgt_spacing: (row_mm, col_mm, z_mm) for target RTDOSE axes
    tgt_shape:   (D, H, W) target shape (RTDOSE frame ordering)
    Returns:     (D, H, W) float32 array in Gy
    """
    # Reorder from GDPHMM (H, W, D) to RTDOSE (D, H, W)
    dose_dhw = np.transpose(dose_hwz, (2, 0, 1)).astype(np.float32)

    sx, sy, sz = src_spacing   # row_mm, col_mm, z_mm  (GDPHMM)
    rx, ry, rz = tgt_spacing   # row_mm, col_mm, z_mm  (RTDOSE)
    zoom = (sz / rz, sx / rx, sy / ry)
    dose_resampled = scipy.ndimage.zoom(dose_dhw, zoom, order=1)

    # Crop or zero-pad to exact target shape (handles rounding differences)
    result = np.zeros(tgt_shape, dtype=np.float32)
    d = min(dose_resampled.shape[0], tgt_shape[0])
    h = min(dose_resampled.shape[1], tgt_shape[1])
    w = min(dose_resampled.shape[2], tgt_shape[2])
    result[:d, :h, :w] = dose_resampled[:d, :h, :w]
    return result


def write_rtdose_dicom(
    template_ds: pydicom.Dataset,
    dose_dhw: np.ndarray,
    output_path: Path,
) -> None:
    """
    Clone template_ds (original RTDOSE header), replace pixel data with
    dose_dhw (Gy, shape D×H×W), and save to output_path.
    All patient demographics and coordinate metadata are kept from template_ds.
    """
    ds = copy.deepcopy(template_ds)

    max_dose = float(dose_dhw.max())
    scale = max_dose / (2**32 - 1) if max_dose > 0 else 1.0
    pixels = (dose_dhw / scale).astype(np.uint32)

    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseGridScaling = scale
    ds.NumberOfFrames = pixels.shape[0]
    ds.Rows = pixels.shape[1]
    ds.Columns = pixels.shape[2]
    ds.PixelData = pixels.tobytes()
    # New SOP Instance UID so this is a distinct DICOM object
    ds.SOPInstanceUID = pydicom.uid.generate_uid()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))


# ---------------------------------------------------------------------------
# File-copy helper
# ---------------------------------------------------------------------------

def copy_dicom_files(src_files: list[Path], dst_dir: Path) -> None:
    """Copy a list of DICOM files into dst_dir, preserving filenames."""
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
    Returns the dict or None if not found.
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
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient_predicted(
    npy_path: Path,
    gdphmm_dir: Path,
    dicom_root: Path,
    output_dir: Path,
    model_name: str,
    pad_size: tuple[int, int],
    gdphmm_spacing: tuple[float, float, float],
) -> bool:
    """Convert one predicted dose .npy to RTDOSE DICOM and copy ancillary files."""
    sample_name = npy_path.stem
    patient_id = sample_name.split('+')[0]
    prefix = f'  [{patient_id}]'

    # 1. Load + denormalise
    dose_raw = np.load(str(npy_path))          # (1, H_pad, W_pad, D) or (H_pad, W_pad, D)
    if dose_raw.ndim == 4:
        dose_raw = dose_raw[0]                  # drop channel dim → (H_pad, W_pad, D)
    dose_padded = dose_raw * GDPHMM_DOSE_MAX   # → Gy

    # 2. Reverse padding
    vol = load_gdphmm_vol(gdphmm_dir, sample_name)
    if vol is None:
        print(f'{prefix} original .npz not found — skipping')
        return False
    orig_h, orig_w = vol['img'].shape[:2]
    dose_gy = reverse_padding(dose_padded, orig_h, orig_w, pad_size)  # (H, W, D)

    # 3. Find patient DICOM directory
    patient_dir = find_patient_dicom_dir(dicom_root, patient_id)
    if patient_dir is None:
        print(f'{prefix} no DICOM directory found — skipping')
        return False
    modality_files = find_dicom_files_by_modality(patient_dir)
    if 'RTDOSE' not in modality_files:
        print(f'{prefix} no RTDOSE file found in {patient_dir} — skipping')
        return False

    # 4. Resample to original RTDOSE grid
    template_ds = pydicom.dcmread(str(modality_files['RTDOSE'][0]))
    tgt_shape, tgt_spacing = get_rtdose_grid(template_ds)
    dose_resampled = resample_dose(dose_gy, gdphmm_spacing, tgt_spacing, tgt_shape)

    # 5. Write new RTDOSE DICOM
    out_dir = output_dir / model_name / patient_id
    write_rtdose_dicom(
        template_ds, dose_resampled,
        out_dir / 'RTDOSE' / f'RD.{patient_id}.dcm',
    )

    # 6. Copy CT, RTSTRUCT, RTPLAN
    for modality, subdir in [('CT', 'CT'), ('RTSTRUCT', 'RTSTRUCT'), ('RTPLAN', 'RTPLAN')]:
        if modality in modality_files:
            copy_dicom_files(modality_files[modality], out_dir / subdir)

    counts = {m: len(v) for m, v in modality_files.items()}
    print(
        f'{prefix} done  '
        f'dose {dose_gy.shape}→{tgt_shape}  '
        f'CT={counts.get("CT", 0)}  '
        f'RS={counts.get("RTSTRUCT", 0)}  '
        f'RP={counts.get("RTPLAN", 0)}'
    )
    return True


def process_patient_original(
    npz_path: Path,
    dicom_root: Path,
    output_dir: Path,
    model_name: str,
) -> bool:
    """Copy original RTDOSE + ancillary DICOM files for one patient."""
    sample_name = npz_path.stem
    patient_id = sample_name.split('+')[0]
    prefix = f'  [{patient_id}]'

    patient_dir = find_patient_dicom_dir(dicom_root, patient_id)
    if patient_dir is None:
        print(f'{prefix} no DICOM directory found — skipping')
        return False

    modality_files = find_dicom_files_by_modality(patient_dir)
    out_dir = output_dir / model_name / patient_id

    for modality, subdir in [
        ('RTDOSE', 'RTDOSE'),
        ('CT', 'CT'),
        ('RTSTRUCT', 'RTSTRUCT'),
        ('RTPLAN', 'RTPLAN'),
    ]:
        if modality in modality_files:
            copy_dicom_files(modality_files[modality], out_dir / subdir)

    counts = {m: len(v) for m, v in modality_files.items()}
    print(
        f'{prefix} done  '
        f'RD={counts.get("RTDOSE", 0)}  '
        f'CT={counts.get("CT", 0)}  '
        f'RS={counts.get("RTSTRUCT", 0)}  '
        f'RP={counts.get("RTPLAN", 0)}'
    )
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert predicted dose .npy → DICOM RTDose, or copy original RTDOSE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--samples-dir', type=Path,
        help='Directory with combined .npy prediction files (required unless --original)',
    )
    parser.add_argument(
        '--model-name', required=True,
        help='Label for this model — used as the output subdirectory name',
    )
    parser.add_argument(
        '--dicom-root', type=Path, required=True, default=Path('/storage/research/artorg_mia/Head_and_Neck'),
        help='Root of the TCIA DICOM data (contains TCIA1_…, TCIA2_…, … subdirectories)',
    )
    parser.add_argument(
        '--output-dir', type=Path, required=True, default=Path('dicom/'),
        help='Output root directory',
    )
    parser.add_argument(
        '--gdphmm-dir', type=Path, required=True, default=Path('data/gdphmm-hn/test'),
        help='GDPHMM split directory containing the original .npz volume files '
             '(e.g. data/gdphmm-hn/test/)',
    )
    parser.add_argument(
        '--pad-size', type=int, nargs=2, default=[176, 160], metavar=('H', 'W'),
        help='H×W padding target used during preprocessing (default: 176 160)',
    )
    parser.add_argument(
        '--voxel-spacing', type=float, nargs=3, default=[2.5, 2.5, 2.0],
        metavar=('ROW', 'COL', 'Z'),
        help='GDPHMM voxel spacing in mm along the H, W, D axes (default: 2.5 2.5 2.0)',
    )
    parser.add_argument(
        '--original', action='store_true',
        help='Copy original RTDOSE DICOM as-is instead of converting .npy predictions',
    )
    parser.add_argument(
        '--file-glob', default='*.npy',
        help='Glob pattern for selecting prediction files in --samples-dir '
             '(default: *.npy).  Use e.g. "*_sigma.npy" to select a specific group.',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.original:
        npz_files = sorted(args.gdphmm_dir.glob('*.npz'))
        if not npz_files:
            raise SystemExit(f'No .npz files found in {args.gdphmm_dir}')
        print(f'Original mode: {len(npz_files)} patients → {args.output_dir / args.model_name}')
        successes = sum(
            process_patient_original(p, args.dicom_root, args.output_dir, args.model_name)
            for p in tqdm(npz_files, desc='Patients', leave=False)
        )
        print(f'\nDone: {successes}/{len(npz_files)} patients copied successfully.')
    else:
        if args.samples_dir is None:
            parser.error('--samples-dir is required when not using --original')
        npy_files = sorted(args.samples_dir.glob(args.file_glob))
        if not npy_files:
            raise SystemExit(f'No files matching "{args.file_glob}" found in {args.samples_dir}')
        print(f'Predicted mode: {len(npy_files)} file(s) → {args.output_dir / args.model_name}')
        gdphmm_spacing = tuple(args.voxel_spacing)
        pad_size = tuple(args.pad_size)
        successes = sum(
            process_patient_predicted(
                p, args.gdphmm_dir, args.dicom_root, args.output_dir,
                args.model_name, pad_size, gdphmm_spacing,
            )
            for p in tqdm(npy_files, desc='Patients', leave=False)
        )
        print(f'\nDone: {successes}/{len(npy_files)} patients converted successfully.')


if __name__ == '__main__':
    main()
