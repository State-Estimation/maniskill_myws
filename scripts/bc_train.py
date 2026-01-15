#!/usr/bin/env python
"""
Behavior Cloning (BC) training on ManiSkill trajectory files recorded by RecordEpisode.

This is intentionally minimal (single-file MLP) to keep the workspace easy to extend.

Example:
  python scripts/bc_train.py --h5 data/demos/debug/20260115_123000.h5 --out outputs/bc_ckpt.pt
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _h5_to_numpy(x: Any) -> np.ndarray:
    import h5py

    if isinstance(x, h5py.Dataset):
        return np.asarray(x)
    raise TypeError(f"Unsupported h5 object: {type(x)}")


def _flatten_h5_group(group) -> np.ndarray:
    """
    Recursively read an h5 group (possibly nested) into a single float32 array by concatenating leaves.
    Leaves are expected to be numeric datasets.
    """
    import h5py

    leaves = []

    def rec(g):
        for k in g.keys():
            v = g[k]
            if isinstance(v, h5py.Group):
                rec(v)
            else:
                arr = np.asarray(v)
                leaves.append(arr)

    rec(group)
    if not leaves:
        raise ValueError("No datasets found under this group")
    # All leaves share leading dims [T(+1), ...]. Flatten trailing dims and concat on last dim.
    flat = [x.reshape(x.shape[0], -1) for x in leaves]
    return np.concatenate(flat, axis=-1).astype(np.float32)


def _load_traj_xy(h5_path: Path):
    import h5py

    Xs, Ys = [], []
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        if not traj_keys:
            raise ValueError(f"No trajectories found in {h5_path}")
        for tk in traj_keys:
            g = f[tk]
            obs_g = g["obs"]
            act_g = g["actions"]
            # obs: [T+1,...]  actions: [T,...]
            if hasattr(obs_g, "keys"):  # group
                obs = _flatten_h5_group(obs_g)
            else:
                obs = np.asarray(obs_g).reshape(np.asarray(obs_g).shape[0], -1).astype(np.float32)
            if hasattr(act_g, "keys"):
                acts = _flatten_h5_group(act_g)
            else:
                acts = np.asarray(act_g).reshape(np.asarray(act_g).shape[0], -1).astype(np.float32)
            T = acts.shape[0]
            Xs.append(obs[:T])
            Ys.append(acts)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True, help="Path to RecordEpisode .h5 file")
    parser.add_argument("--out", type=str, default="outputs/bc_ckpt.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise RuntimeError(
            "BC training requires PyTorch. Install torch in your environment first."
        ) from e

    h5_path = Path(args.h5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, Y = _load_traj_xy(h5_path)
    print("Loaded dataset:", X.shape, Y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, Y.shape[1]),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    N = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred = model(X_t[idx])
            loss = loss_fn(pred, Y_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(idx.shape[0])
        print(f"epoch {ep:03d} loss={total / N:.6f}")

    torch.save(
        dict(
            state_dict=model.state_dict(),
            obs_dim=X.shape[1],
            act_dim=Y.shape[1],
            hidden=args.hidden,
        ),
        out_path,
    )
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()


