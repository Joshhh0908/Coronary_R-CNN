# vis_raw_once.py
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_volume_any(path: str):
    """
    Loads a 3D volume from common formats and returns a numpy array.
    Supported:
      - .npy
      - .npz (uses 'arr_0' or the first key)
      - .pt / .pth (torch tensor or dict containing a tensor/array)
      - .nii / .nii.gz (requires nibabel)
    """
    path = os.path.expanduser(path)
    ext = path.lower()

    if ext.endswith(".npy"):
        vol = np.load(path)

    elif ext.endswith(".npz"):
        z = np.load(path)
        if "arr_0" in z:
            vol = z["arr_0"]
        else:
            key = list(z.keys())[0]
            vol = z[key]

    elif ext.endswith(".pt") or ext.endswith(".pth"):
        obj = torch.load(path, map_location="cpu")
        # common cases: tensor directly, dict with 'x' / 'vol' / 'image'
        if torch.is_tensor(obj):
            vol = obj
        elif isinstance(obj, dict):
            # try common keys
            for k in ["x", "vol", "volume", "image", "img", "data"]:
                if k in obj:
                    vol = obj[k]
                    break
            else:
                # fallback: first tensor-like value
                vol = None
                for v in obj.values():
                    if torch.is_tensor(v):
                        vol = v
                        break
                if vol is None:
                    raise ValueError(f"Loaded dict from {path}, but found no tensor-like value.")
        else:
            raise ValueError(f"Unsupported torch object type: {type(obj)}")
        vol = vol.detach().cpu().numpy() if torch.is_tensor(vol) else np.asarray(vol)

    elif ext.endswith(".nii") or ext.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError(
                "nibabel is required to read .nii/.nii.gz. Install with: pip install nibabel"
            ) from e
        img = nib.load(path)
        vol = img.get_fdata(dtype=np.float32)

    else:
        raise ValueError(f"Unsupported file extension for: {path}")

    vol = np.asarray(vol)
    return vol


def squeeze_to_dhw(vol: np.ndarray) -> np.ndarray:
    """
    Accepts shapes like:
      [D,H,W], [1,D,H,W], [1,1,D,H,W], [H,W,D], etc.
    Tries to end up as [D,H,W].
    """
    v = np.asarray(vol)

    # drop singleton dims
    v = np.squeeze(v)

    if v.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze, got shape {v.shape}")

    # Heuristic: if last dim looks like slices and first dim looks like H
    # You can override with --force_axis_order if needed, but usually dataset is [D,H,W].
    # If D seems to be the smallest dimension, assume it's D.
    dims = v.shape
    d_guess = int(np.argmin(dims))  # smallest dimension
    if d_guess != 0:
        # move that axis to front as D
        v = np.moveaxis(v, d_guess, 0)

    return v  # [D,H,W]


def robust_window(vol: np.ndarray, lo=1.0, hi=99.0):
    """Percentile clip for display."""
    a = vol.astype(np.float32)
    vmin, vmax = np.percentile(a, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    a = np.clip(a, vmin, vmax)
    # normalize to [0,1] for display
    a = (a - vmin) / (vmax - vmin)
    return a, float(vmin), float(vmax)


def save_preview_png(vol_dhw: np.ndarray, out_png: str, slice_idx=None, lo=1.0, hi=99.0):
    """
    Saves a 2x2 preview:
      - axial slice (D index)
      - coronal mid slice
      - sagittal mid slice
      - mean projection along H (D x W)
    """
    D, H, W = vol_dhw.shape
    if slice_idx is None:
        slice_idx = D // 2
    slice_idx = int(np.clip(slice_idx, 0, D - 1))

    disp, vmin, vmax = robust_window(vol_dhw, lo=lo, hi=hi)

    axial = disp[slice_idx]                 # [H,W]
    cor = disp[:, H // 2, :]                # [D,W]
    sag = disp[:, :, W // 2]                # [D,H]
    proj_dw = disp.mean(axis=1)             # mean over H -> [D,W]

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(axial, cmap="gray", interpolation="nearest")
    ax1.set_title(f"Axial slice D={slice_idx}  (H x W)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cor.T, cmap="gray", aspect="auto", origin="lower", interpolation="nearest")
    ax2.set_title("Coronal mid (D x W)")
    ax2.set_xlabel("D")
    ax2.set_ylabel("W")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(sag.T, cmap="gray", aspect="auto", origin="lower", interpolation="nearest")
    ax3.set_title("Sagittal mid (D x H)")
    ax3.set_xlabel("D")
    ax3.set_ylabel("H")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(proj_dw.T, cmap="gray", aspect="auto", origin="lower", interpolation="nearest")
    ax4.set_title("Mean projection over H (D x W)")
    ax4.set_xlabel("D")
    ax4.set_ylabel("W")

    fig.suptitle(f"robust window: p{lo:.0f}..p{hi:.0f}  (raw vmin={vmin:.3g}, vmax={vmax:.3g})", y=0.98)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, type=str, help="Path to raw/cached volume file")
    ap.add_argument("--out", default="raw_preview.png", type=str, help="Output PNG path")
    ap.add_argument("--slice", default=None, type=int, help="Axial slice index (D axis)")
    ap.add_argument("--lo", default=1.0, type=float, help="Low percentile for display window")
    ap.add_argument("--hi", default=99.0, type=float, help="High percentile for display window")
    ap.add_argument("--no_show", action="store_true", help="Do not open an interactive window")
    args = ap.parse_args()

    vol = load_volume_any(args.path)
    vol = squeeze_to_dhw(vol)

    print("Loaded:", args.path)
    print("Volume shape [D,H,W]:", vol.shape, "dtype:", vol.dtype,
          "min/max:", float(np.min(vol)), float(np.max(vol)))

    save_preview_png(vol, args.out, slice_idx=args.slice, lo=args.lo, hi=args.hi)
    print("Saved preview PNG:", args.out)

    if not args.no_show:
        img = plt.imread(args.out)
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(args.out))
        plt.show()


if __name__ == "__main__":
    main()


