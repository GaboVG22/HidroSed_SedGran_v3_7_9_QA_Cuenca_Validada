import numpy as np

from modules.watershed_morphometry import _coverage_stats, _validate_basin_candidate


def test_coverage_stats_detects_border_truncation():
    valid = np.ones((8, 8), dtype=bool)
    basin = np.zeros((8, 8), dtype=bool)
    basin[0:4, 2:5] = True
    stats = _coverage_stats(basin, valid)
    assert stats["touches_dem_border"] is True
    assert stats["border_touch_cells"] > 0


def test_coverage_stats_detects_nodata_contact():
    valid = np.ones((8, 8), dtype=bool)
    valid[4, 4] = False
    basin = np.zeros((8, 8), dtype=bool)
    basin[2:5, 2:4] = True
    stats = _coverage_stats(basin, valid)
    assert stats["touches_nodata"] is True
    assert stats["nodata_adjacent_cells"] > 0


def test_validation_blocks_truncated_basin():
    valid = np.ones((10, 10), dtype=bool)
    basin = np.zeros((10, 10), dtype=bool)
    basin[0:5, 3:7] = True
    acc = np.ones((10, 10), dtype=float)
    acc[4, 5] = 100.0
    qa = _validate_basin_candidate(
        basin=basin,
        valid=valid,
        area_km2=0.02,
        expected_area_km2=None,
        max_area_km2=1.0,
        r0=4,
        c0=4,
        r1=4,
        c1=5,
        acc=acc,
        snapped_dist_m=10.0,
        snap_radius_m=100.0,
        cell_m=10.0,
        decim=1,
    )
    assert qa["cuenca_validada"] is False
    assert qa["controles_minimos"]["cuenca_no_toca_borde_dem"] is False
    assert any("borde del DEM" in d["mensaje"] for d in qa["diagnostico_tecnico"])


def test_validation_accepts_complete_candidate():
    valid = np.ones((20, 20), dtype=bool)
    basin = np.zeros((20, 20), dtype=bool)
    basin[6:14, 7:15] = True
    acc = np.ones((20, 20), dtype=float)
    acc[12, 12] = 250.0
    qa = _validate_basin_candidate(
        basin=basin,
        valid=valid,
        area_km2=0.064,
        expected_area_km2=0.06,
        max_area_km2=1.0,
        r0=12,
        c0=11,
        r1=12,
        c1=12,
        acc=acc,
        snapped_dist_m=10.0,
        snap_radius_m=100.0,
        cell_m=10.0,
        decim=1,
    )
    assert qa["cuenca_validada"] is True
    assert qa["controles_minimos"]["cuenca_no_toca_borde_dem"] is True
    assert qa["controles_minimos"]["cuenca_no_toca_nodata"] is True
