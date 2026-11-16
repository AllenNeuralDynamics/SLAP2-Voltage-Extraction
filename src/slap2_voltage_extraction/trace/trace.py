"""Integration-ROI trace construction and processing."""

from __future__ import annotations

from concurrent.futures import Executor, Future
import numpy as np

from .trace_pixel import TracePixel


def _dmd_dimensions(multi_data_files: object) -> tuple[int, int]:
    header = multi_data_files.header  # type: ignore[attr-defined]
    return int(header["dmdPixelsPerRow"]), int(header["dmdPixelsPerColumn"])


def _normalize_pixel_replacement_map(value: object) -> np.ndarray:
    """Return a replacement map in the ``slap2_utils`` canonical ``(2, N)`` shape."""
    replacement = np.asarray(value, dtype=np.uint32).copy()
    if replacement.size == 0:
        return np.zeros((2, 0), dtype=np.uint32)
    if replacement.ndim == 1:
        if replacement.size % 2:
            raise ValueError(
                "A one-dimensional zPixelReplacementMaps entry must have even length"
            )
        return replacement.reshape(2, -1)
    if replacement.ndim != 2:
        raise ValueError(
            "zPixelReplacementMaps entries must be one- or two-dimensional"
        )
    if replacement.shape[0] == 2:
        return replacement
    if replacement.shape[1] == 2:
        return replacement.T.copy()
    raise ValueError(
        "zPixelReplacementMaps entries must have shape (2, N) or (N, 2)"
    )


class Trace:
    def __init__(self, multi_data_files: object, z_idx: int = 0, ch_idx: int = 0) -> None:
        self.multi_data_files = multi_data_files
        self.z_idx = int(z_idx)
        self.ch_idx = int(ch_idx)
        self.trace_pixels: list[TracePixel] = []
        self.pixel_idxs = np.empty(0, dtype=np.uint32)

    @property
    def super_pixel_ids(self) -> np.ndarray:
        return np.asarray([pixel.super_pixel_id for pixel in self.trace_pixels], dtype=np.uint32)

    def _logical_indices(self, value: object, integration: bool) -> np.ndarray:
        array = np.asarray(value)
        rows, columns = _dmd_dimensions(self.multi_data_files)
        # MATLAB lineage: accept masks whose row and column axes arrive transposed.
        if array.shape == (columns, rows) and columns != rows:
            array = array.T
        if array.shape != (rows, columns):
            raise AssertionError(f"Map must have shape ({rows}, {columns})")
        indices = np.flatnonzero(array.reshape(-1, order="C"))
        if integration:
            indices += array.size
        return indices.astype(np.uint32)

    @staticmethod
    def _numeric_indices(value: object) -> np.ndarray:
        array = np.asarray(value)
        if array.size == 0:
            return np.empty(0, dtype=np.uint32)
        if not np.issubdtype(array.dtype, np.number) or np.any(~np.isfinite(array)) or np.any(array != np.round(array)):
            raise ValueError("Pixel indices must be finite integers")
        if np.any(array < 0):
            raise ValueError("Pixel indices must be nonnegative")
        return array.astype(np.uint32).reshape(-1)

    def set_pixel_idxs(self, raster_pixels: object = (), integration_pixels: object = ()) -> None:
        raster_array = np.asarray(raster_pixels)
        integration_array = np.asarray(integration_pixels)
        raster = (
            self._logical_indices(raster_array, False)
            if raster_array.dtype == bool and raster_array.size
            else self._numeric_indices(raster_pixels)
        )
        integration = (
            self._logical_indices(integration_array, True)
            if integration_array.dtype == bool and integration_array.size
            else self._numeric_indices(integration_pixels)
        )
        self.pixel_idxs = np.concatenate((raster, integration)).astype(np.uint32)
        self.trace_pixels = self._get_trace_pixels(self.pixel_idxs)

    def _get_trace_pixels(self, pixel_idxs: np.ndarray) -> list[TracePixel]:
        requested = np.unique(pixel_idxs.astype(np.uint32))
        replacement = _normalize_pixel_replacement_map(
            self.multi_data_files.hDataFiles[0].zPixelReplacementMaps[  # type: ignore[attr-defined]
                self.z_idx
            ]
        )
        header = self.multi_data_files.header  # type: ignore[attr-defined]
        rows, columns = _dmd_dimensions(self.multi_data_files)
        dmd_pixels = rows * columns
        integration = replacement[1, :] >= dmd_pixels
        replacement[0, integration] += dmd_pixels
        by_pixel = dict(
            zip(replacement[0, :].tolist(), replacement[1, :].tolist(), strict=True)
        )
        mapped = [by_pixel[int(index)] for index in requested if int(index) in by_pixel]
        if not mapped:
            return []
        super_ids, selected_counts = np.unique(mapped, return_counts=True)
        all_ids, all_counts = np.unique(replacement[1, :], return_counts=True)
        full_counts = dict(zip(all_ids.tolist(), all_counts.tolist(), strict=True))
        files = [
            str(item.filename)
            for item in self.multi_data_files.hDataFiles  # type: ignore[attr-defined]
        ]
        pixels = [
            TracePixel(
                file_names=files,
                first_cycle_offset_bytes=int(header["firstCycleOffsetBytes"]),
                bytes_per_cycle=int(header["bytesPerCycle"]),
                lines_per_cycle=int(header["linesPerCycle"]),
                super_pixel_id=int(super_id),
                super_pixel_num_pixels=np.float32(full_counts[int(super_id)]),
                super_pixel_num_pixels_selected=np.float32(selected),
            )
            for super_id, selected in zip(super_ids, selected_counts, strict=True)
        ]
        by_id = {pixel.super_pixel_id: pixel for pixel in pixels}
        line_super_pixel_ids = self.multi_data_files.lineSuperPixelIDs  # type: ignore[attr-defined]
        line_fast_z_idxs = self.multi_data_files.lineFastZIdxs  # type: ignore[attr-defined]
        line_data_start_idxs = self.multi_data_files.lineDataStartIdxs  # type: ignore[attr-defined]
        for line_index, line_ids in enumerate(line_super_pixel_ids):
            if int(line_fast_z_idxs[line_index]) != self.z_idx:
                continue
            ids = np.asarray(line_ids, dtype=np.uint32).reshape(-1)
            for super_id in super_ids:
                matches = np.flatnonzero(ids == super_id)
                if not matches.size:
                    continue
                offset = int(matches[0]) * 2
                offset += ids.size * 2 * self.ch_idx
                # MATLAB lineage: line_data_start_idxs retains the reader's one-based int16 position.
                offset += int(line_data_start_idxs[line_index]) * 2
                offset -= int(header["firstCycleOffsetBytes"])
                pixel = by_id[int(super_id)]
                pixel.byte_offsets = np.append(pixel.byte_offsets, np.uint32(offset))
                pixel.line_idxs = np.append(pixel.line_idxs, np.uint32(line_index))
        return sorted(pixels, key=lambda pixel: pixel.super_pixel_id)

    def load_data(self) -> None:
        for pixel in self.trace_pixels:
            pixel.load()

    def process(self, window_width_lines: int, expected_window_width_lines: int) -> tuple[np.ndarray, np.ndarray]:
        trace, weight, _, _ = TracePixel.process(
            self.trace_pixels, window_width_lines, expected_window_width_lines
        )
        return trace, weight

    def process_async(
        self,
        window_width_lines: int,
        expected_window_width_lines: int,
        *,
        executor: Executor | None = None,
    ) -> Future[tuple[np.ndarray, np.ndarray]]:
        """Schedule trace processing and return a future for ``(trace, weight)``."""

        pixel_future = TracePixel.process_async(
            self.trace_pixels,
            window_width_lines,
            expected_window_width_lines,
            executor=executor,
        )
        result: Future[tuple[np.ndarray, np.ndarray]] = Future()

        def finalize(completed: Future[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
            try:
                trace, weight, _, _ = completed.result()
                result.set_result((trace, weight))
            except BaseException as exc:
                result.set_exception(exc)

        pixel_future.add_done_callback(finalize)
        return result


__all__ = ["Trace"]
