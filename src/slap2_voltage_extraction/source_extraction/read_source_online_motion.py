from __future__ import annotations

import numpy as np


def read_source_online_motion(
    multi_data_files: object,
    samples_per_cycle: int = 1,
    line_rate_hz: float | None = None,
) -> dict[str, object]:
    """Read uniformly sampled online motion offsets from SLAP2 line headers.

    ``multi_data_files`` must expose canonical ``slap2_utils`` line-header
    indices, cycle count, and ``getOnlineMotion`` access. The reader receives
    one-based global line IDs; this package converts them to zero-based IDs at
    its internal boundary.
    """

    if (
        isinstance(samples_per_cycle, (bool, np.bool_))
        or not isinstance(samples_per_cycle, (int, np.integer))
        or samples_per_cycle <= 0
    ):
        raise ValueError("samples_per_cycle must be a positive integer")
    if (
        line_rate_hz is None
        or isinstance(line_rate_hz, (bool, np.bool_))
        or not np.isscalar(line_rate_hz)
        or not np.isreal(line_rate_hz)
        or not np.isfinite(line_rate_hz)
        or float(line_rate_hz) <= 0
    ):
        raise ValueError("line_rate_hz must be a positive finite real scalar")
    rate = float(line_rate_hz)

    header_indices = multi_data_files.lineHeaderIdxs  # type: ignore[attr-defined]
    lines_per_cycle = len(header_indices)
    num_cycles = int(multi_data_files.numCycles)  # type: ignore[attr-defined]
    sample_count = min(int(samples_per_cycle), lines_per_cycle)
    within_cycle = np.unique(
        np.floor(np.arange(sample_count) * lines_per_cycle / sample_count).astype(np.int64)
    )
    line_ids = (
        within_cycle[:, None]
        + np.arange(num_cycles, dtype=np.int64)[None, :] * lines_per_cycle
    ).reshape(-1, order="F")

    reader = getattr(multi_data_files, "getOnlineMotion", None)
    if reader is None:
        raise AttributeError(
            "The SLAP2 reader does not support online-motion access; "
            "online motion censoring is unavailable with this backend."
        )
    x, y, z = reader(line_ids + 1)

    return {
        "x": np.asarray(x).reshape(-1),
        "y": np.asarray(y).reshape(-1),
        "z": np.asarray(z).reshape(-1),
        "line_ids": line_ids,
        "fs_offsets": rate / lines_per_cycle * within_cycle.size,
        "lines_per_cycle": lines_per_cycle,
        "num_cycles": num_cycles,
        "total_lines": lines_per_cycle * num_cycles,
        "line_rate_hz": rate,
        "sample_times_s": line_ids / rate,
    }


__all__ = ["read_source_online_motion"]
