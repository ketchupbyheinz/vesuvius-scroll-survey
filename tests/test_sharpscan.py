import json

import numpy as np
import tifffile

from vesuvius_survey import render_patch, sharpscan


def test_gap_ratio_flags_ink_on_gaps_and_not_shifted_copies():
    rng = np.random.default_rng(0)
    H = W = 200
    gap = rng.random((H, W)) * 0.2
    gap[60:90, 60:90] = 0.9                     # an air gap
    valid = np.ones((H, W), bool)
    on_gap = np.zeros((H, W), bool)
    on_gap[65:85, 65:85] = True
    r = sharpscan.gap_ratio(gap, on_gap, valid)
    sh = sharpscan.shifted_ratios(gap, on_gap, valid)
    assert r > 3
    assert len(sh) == 48
    assert np.median(sh) < 1.2
    assert all(r > s for s in sh)
    elsewhere = np.zeros((H, W), bool)
    elsewhere[150:170, 150:170] = True
    assert sharpscan.gap_ratio(gap, elsewhere, valid) < 1


def test_gap_fraction_uses_each_layers_darkest_quartile():
    stack = np.full((2, 4, 4), 200, np.uint8)
    stack[:, 0, 0] = 10
    stack[0, 1, 1] = 10
    g = sharpscan.gap_fraction(stack, range(2))
    assert g[0, 0] == 1.0 and g[1, 1] == 0.5 and g[2, 2] == 0.0


def test_map_mesh_scales_offsets_and_keeps_invalid(tmp_path):
    src, dst = tmp_path / "m", tmp_path / "o"
    src.mkdir()
    for c, v in zip("xyz", (100.0, 200.0, 300.0)):
        a = np.full((3, 3), v, np.float32)
        a[0, 0] = -1
        tifffile.imwrite(src / f"{c}.tif", a)
    (src / "meta.json").write_text(json.dumps({"scale": [0.05, 0.05]}))
    sharpscan.map_mesh(src, dst, k=4.0, offset=(2.0, -2.0, -100.0), level=1)
    x, z = tifffile.imread(dst / "x.tif"), tifffile.imread(dst / "z.tif")
    assert x[1, 1] == (100 * 4 + 2) / 2 and z[1, 1] == (300 * 4 - 100) / 2
    assert x[0, 0] == -1 and z[0, 0] == -1
    meta = json.loads((dst / "meta.json").read_text())
    assert meta["scale"] == [0.025, 0.025] and meta["sharpscan_mapping"]["level"] == 1


def _tiny_scene(tmp_path):
    import zarr
    rng = np.random.default_rng(3)
    g = zarr.open_group(str(tmp_path / "ct.zarr"), mode="w", zarr_format=2)
    ct = g.create_array("0", shape=(40, 40, 40), chunks=(16, 16, 16), dtype="u1")
    ct[:] = rng.integers(1, 255, (40, 40, 40), dtype=np.uint8)
    m = tmp_path / "mesh"
    m.mkdir()
    yy, xx = np.mgrid[0:12, 0:12].astype(np.float32)
    for c, a in zip("xyz", (5 + xx * 2.5, 5 + yy * 2.5, np.full_like(xx, 20))):
        tifffile.imwrite(m / f"{c}.tif", a)
    (m / "meta.json").write_text(json.dumps({"scale": [0.1, 0.1]}))
    return m, str(tmp_path / "ct.zarr")


def test_render_is_identical_with_many_workers_and_misaligned_tiles(tmp_path):
    import zarr
    mesh, ct = _tiny_scene(tmp_path)
    outs = []
    for w in (1, 4, 4):
        o = tmp_path / f"r{w}_{len(outs)}.zarr"
        render_patch.main([str(mesh), ct, str(o), "--rows", "0", "100", "--cols", "0", "100",
                           "--layers", "3", "--tile", "7", "--workers", str(w)])
        outs.append(np.asarray(zarr.open_array(str(o), mode="r")))
    assert outs[0].any() and all(np.array_equal(outs[0], x) for x in outs[1:])


def test_render_level_and_coord_mult_sample_the_coarser_level(tmp_path):
    """Mesh in level-0 voxels, rendered on level 1 with --coord-mult 0.5: a linear CT gives the analytic value."""
    import zarr
    mesh, _ = _tiny_scene(tmp_path)
    z, y, x = np.mgrid[0:40, 0:40, 0:40]
    g = zarr.open_group(str(tmp_path / "lin.zarr"), mode="w", zarr_format=2)
    g.create_array("0", shape=(40, 40, 40), chunks=(16, 16, 16), dtype="u1")[:] = (x + 2 * y).astype(np.uint8)
    g.create_array("1", shape=(20, 20, 20), chunks=(16, 16, 16), dtype="u1")[:] = (x + 2 * y)[::2, ::2, ::2].astype(np.uint8)
    b = tmp_path / "b.zarr"
    render_patch.main([str(mesh), str(tmp_path / "lin.zarr"), str(b), "--rows", "0", "20", "--cols", "0", "20",
                       "--layers", "1", "--level", "1", "--coord-mult", "0.5"])
    B = np.asarray(zarr.open_array(str(b), mode="r"))[0]
    r, c = np.mgrid[0:20, 0:20]
    u, v = (c + 0.5) * 0.2, (r + 0.5) * 0.2            # mesh node coordinates of each pixel (scale 0.1 / 0.5)
    want = np.floor((5 + 2.5 * u) + 2 * (5 + 2.5 * v))
    inside = (u >= 1) & (u <= 10) & (v >= 1) & (v <= 10)   # border nodes have no normal and are not drawn
    assert B.shape == (20, 20) and np.abs(B[inside] - want[inside]).max() <= 1
    assert zarr.open_array(str(b), mode="r").attrs["level"] == "1"


def test_read_box_refuses_boxes_that_leave_the_volume(tmp_path):
    import pytest
    _, ct = _tiny_scene(tmp_path)
    assert sharpscan.read_box(ct, 0, 0, 4, 0, 4, 0, 4).shape == (4, 4, 4)
    with pytest.raises(ValueError):
        sharpscan.read_box(ct, 0, -2, 4, 0, 4, 0, 4)
    with pytest.raises(ValueError):
        sharpscan.read_box(ct, 0, 0, 4, 0, 4, 38, 42)


def test_resume_with_other_settings_starts_fresh(tmp_path):
    import zarr
    mesh, ct = _tiny_scene(tmp_path)
    g = zarr.open_group(ct, mode="r+")
    g.create_array("1", shape=(20, 20, 20), chunks=(16, 16, 16), dtype="u1")[:] = 7
    o = tmp_path / "r.zarr"
    args = [str(mesh), ct, str(o), "--rows", "20", "60", "--cols", "20", "60", "--layers", "1", "--resume"]
    render_patch.main(args)
    first = np.asarray(zarr.open_array(str(o), mode="r"))
    render_patch.main(args + ["--level", "1", "--coord-mult", "0.5"])
    second = np.asarray(zarr.open_array(str(o), mode="r"))
    assert first.any() and not np.array_equal(first, second) and set(np.unique(second)) <= {0, 7}
