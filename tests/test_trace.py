from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

trace_module = pytest.importorskip(
    "slap2_voltage_extraction.trace",
    reason="Python Trace/TracePixel translation has not been implemented yet",
)
Trace = trace_module.Trace


@dataclass
class DataFile:
    filename: str
    zPixelReplacementMaps: list[np.ndarray]


@dataclass
class FakeMultiDataFiles:
    """Canonical camelCase surface exported by slap2_utils.MultiDataFiles."""

    header: dict[str, int]
    metaData: Any
    hDataFiles: list[DataFile]
    lineSuperPixelIDs: list[np.ndarray]
    lineFastZIdxs: np.ndarray
    lineDataStartIdxs: np.ndarray


def mapping_source() -> FakeMultiDataFiles:
    return FakeMultiDataFiles(
        header={
            "dmdPixelsPerRow": 3,
            "dmdPixelsPerColumn": 2,
            "bytesPerCycle": 64,
            "firstCycleOffsetBytes": 2,
            "linesPerCycle": 4,
        },
        metaData=object(),
        hDataFiles=[
            DataFile(
                "first.dat",
                [
                    np.asarray(
                        [
                            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
                            [1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10],
                        ],
                        dtype=np.uint32,
                    )
                ],
            ),
            DataFile("second.dat", []),
        ],
        lineSuperPixelIDs=[
            np.asarray([1, 2, 6], dtype=np.uint32),
            np.asarray([2, 1], dtype=np.uint32),
            np.asarray([6, 1], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
        ],
        lineFastZIdxs=np.asarray([0, 0, 1, 0]),
        lineDataStartIdxs=np.asarray([1, 10, 20, 30]),
    )


def processing_source(filename: Path) -> FakeMultiDataFiles:
    return FakeMultiDataFiles(
        header={
            "dmdPixelsPerRow": 2,
            "dmdPixelsPerColumn": 2,
            "bytesPerCycle": 8,
            "firstCycleOffsetBytes": 0,
            "linesPerCycle": 2,
        },
        metaData=object(),
        hDataFiles=[
            DataFile(str(filename), [np.asarray([[0], [1]], dtype=np.uint32)])
        ],
        lineSuperPixelIDs=[
            np.asarray([1], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
        ],
        lineFastZIdxs=np.asarray([0, 0]),
        lineDataStartIdxs=np.asarray([1, 3]),
    )


class TestTrace:
    def test_constructor_stores_typed_inputs_and_defaults(self) -> None:
        source = mapping_source()

        trace = Trace(source)

        assert trace.multi_data_files is source
        assert trace.z_idx == 0
        assert trace.ch_idx == 0
        assert np.asarray(trace.pixel_idxs).size == 0
        assert np.asarray(trace.super_pixel_ids).size == 0

    def test_numeric_pixel_lists_create_expected_trace_pixels(self) -> None:
        trace = Trace(mapping_source(), z_idx=0, ch_idx=1)

        trace.set_pixel_idxs([0, 1], [6])

        np.testing.assert_array_equal(trace.pixel_idxs, np.asarray([0, 1, 6], dtype=np.uint32))
        np.testing.assert_array_equal(trace.super_pixel_ids, [1, 6])
        np.testing.assert_array_equal(
            [pixel.super_pixel_num_pixels for pixel in trace.trace_pixels],
            np.asarray([2, 2], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            [pixel.super_pixel_num_pixels_selected for pixel in trace.trace_pixels],
            np.asarray([2, 1], dtype=np.float32),
        )
        np.testing.assert_array_equal(trace.trace_pixels[0].line_idxs, np.asarray([0, 1, 3], dtype=np.uint32))
        np.testing.assert_array_equal(trace.trace_pixels[0].byte_offsets, np.asarray([6, 24, 60], dtype=np.uint32))
        np.testing.assert_array_equal(trace.trace_pixels[1].line_idxs, np.asarray([0], dtype=np.uint32))
        np.testing.assert_array_equal(trace.trace_pixels[1].byte_offsets, np.asarray([10], dtype=np.uint32))
        assert trace.trace_pixels[0].file_names == ["first.dat", "second.dat"]

    def test_logical_maps_and_transposed_map_match_numeric_indices(self) -> None:
        from_maps = Trace(mapping_source())
        from_lists = Trace(mapping_source())
        raster_map = np.zeros((3, 2), dtype=bool)
        raster_map[0, [0, 1]] = True
        integration_map = np.zeros((3, 2), dtype=bool)
        integration_map[0, 0] = True

        from_maps.set_pixel_idxs(raster_map, integration_map.T)
        from_lists.set_pixel_idxs([0, 1], [6])

        np.testing.assert_array_equal(from_maps.pixel_idxs, from_lists.pixel_idxs)
        np.testing.assert_array_equal(from_maps.super_pixel_ids, from_lists.super_pixel_ids)

    def test_incorrect_logical_map_size_fails(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            Trace(mapping_source()).set_pixel_idxs(np.zeros((4, 4), dtype=bool), [])

    def test_noninteger_pixel_list_fails_validation(self) -> None:
        with pytest.raises((TypeError, ValueError)) as caught:
            Trace(mapping_source()).set_pixel_idxs([1.5], [])
        assert "integer" in str(caught.value).lower()

    def test_unknown_pixels_are_retained_as_request_but_not_traces(self) -> None:
        trace = Trace(mapping_source())

        trace.set_pixel_idxs([0, 98], [])

        np.testing.assert_array_equal(trace.pixel_idxs, np.asarray([0, 98], dtype=np.uint32))
        np.testing.assert_array_equal(trace.super_pixel_ids, [1])

    @pytest.mark.parametrize(
        "replacement",
        [
            np.asarray([[0, 1], [1, 1]], dtype=np.uint32),
            np.asarray([0, 1, 1, 1], dtype=np.uint32),
        ],
    )
    def test_supported_replacement_map_representations_are_normalized(
        self, replacement: np.ndarray
    ) -> None:
        source = mapping_source()
        source.hDataFiles[0].zPixelReplacementMaps[0] = replacement

        trace = Trace(source)
        trace.set_pixel_idxs([0, 1], [])

        np.testing.assert_array_equal(trace.super_pixel_ids, [1])
        assert trace.trace_pixels[0].super_pixel_num_pixels == 2

    def test_empty_replacement_map_produces_no_trace_pixels(self) -> None:
        source = mapping_source()
        source.hDataFiles[0].zPixelReplacementMaps[0] = np.empty((0, 0))

        trace = Trace(source)
        trace.set_pixel_idxs([0], [])

        assert not trace.trace_pixels

    @pytest.mark.parametrize(
        "replacement",
        [
            np.asarray([0, 1, 2], dtype=np.uint32),
            np.zeros((3, 3), dtype=np.uint32),
            np.zeros((2, 2, 2), dtype=np.uint32),
        ],
    )
    def test_invalid_replacement_map_shape_fails_clearly(
        self, replacement: np.ndarray
    ) -> None:
        source = mapping_source()
        source.hDataFiles[0].zPixelReplacementMaps[0] = replacement

        with pytest.raises(ValueError, match="zPixelReplacementMaps"):
            Trace(source).set_pixel_idxs([0], [])

    def test_repeated_selection_does_not_duplicate_trace_pixels(self) -> None:
        trace = Trace(mapping_source())

        trace.set_pixel_idxs([0, 1], [6])
        trace.set_pixel_idxs([0, 1], [6])

        np.testing.assert_array_equal(trace.super_pixel_ids, [1, 6])
        assert len(trace.trace_pixels) == 2

    def test_load_data_and_process_delegate_to_trace_pixels(self, write_int16) -> None:
        filename = write_int16("trace.dat", np.full(8, 10, dtype=np.int16))
        trace_object = Trace(processing_source(filename))
        trace_object.set_pixel_idxs([0], [])

        trace_object.load_data()
        pixel = trace_object.trace_pixels[0]
        assert pixel.is_loaded is True
        np.testing.assert_array_equal(pixel.data, np.full((2, 2), 10, dtype=np.float32))

        trace, weight = trace_object.process(1, 1)

        assert pixel.is_loaded is False
        assert pixel.data.dtype == np.float32
        assert pixel.data.shape == (0,)
        assert trace.shape == (4,)
        assert weight.shape == (4,)
        np.testing.assert_allclose(trace, np.full(4, 10, dtype=np.float32), atol=1e-5, rtol=0)
        assert np.all(weight > 0)
