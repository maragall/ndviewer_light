"""Tests for IMA-191 per-region FOV subset (subset_fovs_per_region).

"condition" == region/well: the helper keeps the first N FOVs (by integer fov
index) within each region, preserving original order and handling boundary
inputs.
"""

from ndviewer_light.core import subset_fovs_per_region


def _fov(region, fov):
    return {"region": region, "fov": fov, "path": f"{region}/fov_{fov}"}


def test_disabled_when_n_zero_or_negative():
    fovs = [_fov("A1", 0), _fov("A1", 1)]
    assert subset_fovs_per_region(fovs, 0) == fovs
    assert subset_fovs_per_region(fovs, -5) == fovs


def test_empty_input():
    assert subset_fovs_per_region([], 3) == []


def test_keeps_first_n_per_region():
    fovs = [
        _fov("A1", 0), _fov("A1", 1), _fov("A1", 2),
        _fov("B2", 0), _fov("B2", 1),
    ]
    result = subset_fovs_per_region(fovs, 1)
    assert result == [_fov("A1", 0), _fov("B2", 0)]


def test_region_smaller_than_n_keeps_all():
    fovs = [_fov("A1", 0), _fov("A1", 1), _fov("B2", 0)]
    # n=5 exceeds every region's size -> everything kept, order preserved.
    assert subset_fovs_per_region(fovs, 5) == fovs


def test_lexical_order_trap():
    # Discovery sorts directory names lexically, so 'fov_10' precedes 'fov_2'
    # in the input. The helper must rank by INTEGER fov index.
    fovs = [
        _fov("A1", 0), _fov("A1", 1), _fov("A1", 10), _fov("A1", 11),
        _fov("A1", 2),
    ]
    result = subset_fovs_per_region(fovs, 3)
    kept = [f["fov"] for f in result]
    assert kept == [0, 1, 2], f"expected lowest 3 int indices, got {kept}"


def test_multi_region_mixed_sizes():
    fovs = [
        _fov("A1", 0), _fov("A1", 1), _fov("A1", 2),
        _fov("B2", 0),
        _fov("C3", 0), _fov("C3", 1),
    ]
    result = subset_fovs_per_region(fovs, 2)
    kept = [(f["region"], f["fov"]) for f in result]
    assert kept == [("A1", 0), ("A1", 1), ("B2", 0), ("C3", 0), ("C3", 1)]


def test_preserves_original_order():
    # Interleaved regions in the input keep their original relative positions.
    fovs = [_fov("A1", 0), _fov("B2", 0), _fov("A1", 1), _fov("B2", 1)]
    result = subset_fovs_per_region(fovs, 1)
    assert result == [_fov("A1", 0), _fov("B2", 0)]


def test_single_6d_region_entry():
    # 6D discovery yields one entry per region (fov=0); n>=1 keeps it.
    fovs = [_fov("default", 0)]
    assert subset_fovs_per_region(fovs, 1) == fovs
