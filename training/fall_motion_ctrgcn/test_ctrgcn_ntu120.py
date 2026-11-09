import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def strip_module_prefix(state_dict):
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def _as_dtype(dtype_str_or_np):
    if dtype_str_or_np is None:
        return None
    if isinstance(dtype_str_or_np, np.dtype):
        return dtype_str_or_np
    return np.dtype(dtype_str_or_np)


class NTUFromKMP(Dataset):
    """
    Loads X from a binary memmap file (X.kmp) with shape/dtype described by meta.npy,
    and labels from y.npy.

    Outputs per sample: (C, T, V, M) float32 tensor + int label
    """
    def __init__(self, cfg):
        data_cfg = cfg["data"]
        input_cfg = cfg["input"]
        test_cfg = cfg["test_time"]

        self.y = np.load(data_cfg["y_npy_path"]).astype(np.int64)

        # meta can be dict-like or array; we handle common patterns robustly
        meta = np.load(data_cfg["meta_npy_path"], allow_pickle=True)
        meta_obj = meta.item() if isinstance(meta, np.ndarray) and meta.shape == () else meta

        # Try to infer shape/dtype from meta
        shape = None
        dtype = None

        if isinstance(meta_obj, dict):
            # Common keys
            for k in ["shape", "x_shape", "data_shape"]:
                if k in meta_obj:
                    shape = tuple(meta_obj[k])
                    break
            for k in ["dtype", "x_dtype", "data_dtype"]:
                if k in meta_obj:
                    dtype = np.dtype(meta_obj[k])
                    break

        # YAML overrides
        if input_cfg.get("dtype") is not None:
            dtype = np.dtype(input_cfg["dtype"])

        if shape is None:
            raise ValueError(
                "Could not infer X shape from meta.npy. "
                "Open meta.npy and check what it contains (shape keys)."
            )
        if dtype is None:
            # Default guess; but better to get it from meta
            dtype = np.float32

        self.layout = input_cfg.get("layout", "NCTVM")
        self.fixed_T = test_cfg.get("fixed_T", None)
        self.pad_mode = test_cfg.get("pad_mode", "zeros")
        self.use_bone = test_cfg.get("use_bone", False)
        self.use_velocity = test_cfg.get("use_velocity", False)

        # Memmap X
        self.X = np.memmap(
            data_cfg["x_kmp_path"],
            mode="r",
            dtype=dtype,
            shape=shape
        )

        # Basic sanity
        if len(self.y) != shape[0]:
            raise ValueError(f"Mismatch: y has {len(self.y)} samples but X has {shape[0]} samples.")

    def __len__(self):
        return len(self.y)

    def _to_ctvm(self, x):
        """
        Convert stored layout -> (C,T,V,M)
        Supports:
          - NCTVM: x is already (C,T,V,M)
          - NTVCM: stored as (T,V,C,M) -> permute to (C,T,V,M)
        """
        if self.layout.upper() == "NCTVM":
            # x: (C,T,V,M)
            return x
        if self.layout.upper() == "NTVCM":
            # x: (T,V,C,M) -> (C,T,V,M)
            return np.transpose(x, (2, 0, 1, 3))
        raise ValueError(f"Unsupported layout: {self.layout}. Use NCTVM or NTVCM.")

    def _pad_or_clip_T(self, x_ctvm):
        if self.fixed_T is None:
            return x_ctvm
        C, T, V, M = x_ctvm.shape
        if T == self.fixed_T:
            return x_ctvm
        if T > self.fixed_T:
            return x_ctvm[:, :self.fixed_T, :, :]
        pad = self.fixed_T - T
        if self.pad_mode == "repeat_last" and T > 0:
            last = x_ctvm[:, -1:, :, :]
            pad_tensor = np.repeat(last, pad, axis=1)
        else:
            pad_tensor = np.zeros((C, pad, V, M), dtype=x_ctvm.dtype)
        return np.concatenate([x_ctvm, pad_tensor], axis=1)

    def _compute_velocity(self, x_ctvm):
        v = np.zeros_like(x_ctvm)
        v[:, :-1, :, :] = x_ctvm[:, 1:, :, :] - x_ctvm[:, :-1, :, :]
        return v

    def _compute_bone(self, x_ctvm):
        # Standard NTU 25-joint pairs (1-indexed in literature)
        bone_pairs = (
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
            (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
            (11, 10), (12, 11), (13, 1), (14, 13),
            (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (23, 8),
            (24, 25), (25, 12)
        )
        bone = np.zeros_like(x_ctvm)
        for v1, v2 in bone_pairs:
            bone[:, :, v1 - 1, :] = x_ctvm[:, :, v1 - 1, :] - x_ctvm[:, :, v2 - 1, :]
        return bone

    def __getitem__(self, idx):
        x = np.array(self.X[idx], copy=False)  # view into memmap
        x = self._to_ctvm(x).astype(np.float32, copy=False)
        x = self._pad_or_clip_T(x)

        streams = [x]
        if self.use_bone:
            streams.append(self._compute_bone(x))
        if self.use_velocity:
            streams.append(self._compute_velocity(x))

        x = np.concatenate(streams, axis=0)  # (C',T,V,M)
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_model(cfg):
    # Adjust import to match your repo
    from model.ctrgcn import Model

    return Model(
        num_class=cfg["num_class"],
        num_point=cfg["num_point"],
        num_person=cfg["num_person"],
        graph=cfg["model"]["graph"],
        graph_args=cfg["model"]["graph_args"],
    )


@torch.no_grad()
def evaluate(model, loader, device, topk_list):
    model.eval()
    correct = {k: 0 for k in topk_list}
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        for k in topk_list:
            k_eff = min(k, logits.shape[1])
            topk = torch.topk(logits, k=k_eff, dim=1).indices
            correct[k] += (topk == y.unsqueeze(1)).any(dim=1).sum().item()

        total += y.numel()

    return {k: 100.0 * correct[k] / total for k in topk_list}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"].startswith("cuda") else "cpu")
    print("Device:", device)

    # Dataset selection
    if cfg["data"]["format"].lower() != "kmp":
        raise ValueError("This script is for data.format='kmp'. (X.kmp + y.npy + meta.npy)")

    ds = NTUFromKMP(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg.get("pin_memory", device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(cfg).to(device)

    ckpt = torch.load(cfg["ckpt_path"], map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Checkpoint loaded.")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")

    results = evaluate(model, loader, device, cfg["metrics"]["topk"])
    print(f"\nNTU120 results ({cfg['data'].get('split','?')}):")
    for k, v in results.items():
        print(f"Top-{k} Accuracy: {v:.2f}%")


if __name__ == "__main__":
    main()