"""Test bench: routetools.performance vs SWOPP3 reference.

Compares our closed-form parametric model (``routetools.performance``)
against the compiled SWOPP3 performance model (``predict_no_wps`` and
``predict_with_wps``) across structured grids, edge cases, and random
stress tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from routetools.performance import (
    K_A,
    K_H,
    predict_power,
    predict_power_batch,
    predict_power_no_wps as parametric_no_wps,
    predict_power_with_wps as parametric_with_wps,
)

# ---------------------------------------------------------------------------
# Reference package: only skip tests that actually use SWOPP3
# ---------------------------------------------------------------------------
try:
    import swopp3_performance_model as swopp3  # type: ignore[import-untyped]
except ModuleNotFoundError:
    swopp3 = None  # type: ignore[assignment]

needs_swopp3 = pytest.mark.skipif(
    swopp3 is None,
    reason="swopp3_performance_model wheel is not installed",
)


# ===================================================================
#  predict_no_wps  tests
# ===================================================================
@needs_swopp3
class TestNoWPS:
    """Parametric model vs reference predict_no_wps."""

    # ------ Structured grid ------
    @pytest.mark.parametrize(
        "tws",
        [0, 2, 5, 10, 15, 20, 25, 30],
    )
    @pytest.mark.parametrize(
        "twa",
        [0, 30, 60, 90, 120, 150, 180],
    )
    @pytest.mark.parametrize(
        "v",
        [0, 2, 5, 8, 10, 12, 14],
    )
    def test_structured_grid_calm_sea(self, tws: float, twa: float, v: float) -> None:
        """No waves (swh=0, mwa=0): hull + wind only."""
        ref = swopp3.predict_no_wps(tws, twa, 0.0, 0.0, v)
        par = parametric_no_wps(tws, twa, 0.0, 0.0, v)
        assert abs(par - ref) < 0.15, (
            f"tws={tws}, twa={twa}, v={v}: ref={ref:.4f}, par={par:.4f}, "
            f"err={abs(par - ref):.4f}"
        )

    @pytest.mark.parametrize(
        "swh",
        [0, 1, 2, 4, 6, 8],
    )
    @pytest.mark.parametrize(
        "mwa",
        [0, 45, 90, 135, 180],
    )
    @pytest.mark.parametrize(
        "v",
        [0, 3, 7, 10, 14],
    )
    def test_structured_grid_wave_only(self, swh: float, mwa: float, v: float) -> None:
        """No wind (tws=0): hull + wave only."""
        ref = swopp3.predict_no_wps(0.0, 0.0, swh, mwa, v)
        par = parametric_no_wps(0.0, 0.0, swh, mwa, v)
        assert abs(par - ref) < 0.15, (
            f"swh={swh}, mwa={mwa}, v={v}: ref={ref:.4f}, par={par:.4f}, "
            f"err={abs(par - ref):.4f}"
        )

    @pytest.mark.parametrize(
        "tws,twa,swh,mwa,v",
        [
            (10, 45, 2, 30, 8),
            (15, 90, 3, 90, 10),
            (20, 135, 5, 150, 6),
            (5, 0, 1, 0, 12),
            (25, 180, 4, 180, 4),
            (8, 60, 6, 45, 14),
            (12, 120, 0.5, 60, 9),
            (30, 0, 8, 0, 2),
        ],
    )
    def test_combined_representative(
        self,
        tws: float,
        twa: float,
        swh: float,
        mwa: float,
        v: float,
    ) -> None:
        """Hand-picked combined-condition cases."""
        ref = swopp3.predict_no_wps(tws, twa, swh, mwa, v)
        par = parametric_no_wps(tws, twa, swh, mwa, v)
        assert abs(par - ref) < 0.15, (
            f"tws={tws}, twa={twa}, swh={swh}, mwa={mwa}, v={v}: "
            f"ref={ref:.4f}, par={par:.4f}, err={abs(par - ref):.4f}"
        )

    # ------ Edge cases ------
    def test_all_zeros(self) -> None:
        """Zero inputs should give zero power."""
        ref = swopp3.predict_no_wps(0, 0, 0, 0, 0)
        par = parametric_no_wps(0, 0, 0, 0, 0)
        assert ref == 0.0
        assert par == 0.0

    def test_zero_speed(self) -> None:
        """At v=0, all power terms vanish regardless of environment."""
        for tws in [0, 10, 20, 30]:
            for swh in [0, 3, 8]:
                ref = swopp3.predict_no_wps(tws, 90, swh, 90, 0)
                par = parametric_no_wps(tws, 90, swh, 90, 0)
                assert ref == 0.0
                assert par == 0.0

    def test_twa_symmetry(self) -> None:
        """Power should be symmetric around TWA=0 (same for +/- angles)."""
        for tws, twa, v in [(10, 30, 8), (15, 60, 5), (20, 90, 10)]:
            ref_pos = swopp3.predict_no_wps(tws, twa, 2, 45, v)
            ref_neg = swopp3.predict_no_wps(tws, -twa, 2, 45, v)
            par_pos = parametric_no_wps(tws, twa, 2, 45, v)
            par_neg = parametric_no_wps(tws, -twa, 2, 45, v)
            assert abs(ref_pos - ref_neg) < 1e-10
            assert abs(par_pos - par_neg) < 1e-10

    def test_mwa_symmetry(self) -> None:
        """Power should be symmetric around MWA=0."""
        for mwa in [30, 60, 90, 120, 150]:
            ref_pos = swopp3.predict_no_wps(10, 45, 3, mwa, 8)
            ref_neg = swopp3.predict_no_wps(10, 45, 3, -mwa, 8)
            par_pos = parametric_no_wps(10, 45, 3, mwa, 8)
            par_neg = parametric_no_wps(10, 45, 3, -mwa, 8)
            assert abs(ref_pos - ref_neg) < 1e-10
            assert abs(par_pos - par_neg) < 1e-10

    def test_clamping_at_zero(self) -> None:
        """Strong tailwind should clamp power at zero."""
        ref = swopp3.predict_no_wps(25, 180, 0, 0, 2)
        par = parametric_no_wps(25, 180, 0, 0, 2)
        assert ref == 0.0
        assert par == 0.0

    # ------ Random stress test ------
    def test_random_stress_no_wps(self) -> None:
        """10 000 random inputs: max absolute error < 0.15 kW."""
        rng = np.random.default_rng(42)
        n = 10_000
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        ref_vals = np.array(
            [
                swopp3.predict_no_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
                for i in range(n)
            ]
        )
        par_vals = predict_power_batch(tws, twa, swh, mwa, v, wps=False)
        abs_errs = np.abs(par_vals - ref_vals)

        max_err = abs_errs.max()
        mean_err = abs_errs.mean()
        p99_err = np.percentile(abs_errs, 99)

        # Report
        print(
            f"\n[no_wps random stress] n={n}, "
            f"max_err={max_err:.4f} kW, mean_err={mean_err:.4f} kW, "
            f"p99_err={p99_err:.4f} kW"
        )

        assert max_err < 0.15, f"Max absolute error too large: {max_err:.4f} kW"


# ===================================================================
#  predict_with_wps  tests
# ===================================================================
@needs_swopp3
class TestWithWPS:
    """Parametric model vs reference predict_with_wps."""

    @pytest.mark.parametrize(
        "tws,twa,swh,mwa,v",
        [
            (10, 45, 2, 30, 8),
            (15, 90, 3, 90, 10),
            (20, 135, 5, 150, 6),
            (5, 0, 1, 0, 12),
            (25, 180, 4, 180, 4),
            (8, 60, 6, 45, 14),
            (12, 120, 0.5, 60, 9),
            (30, 0, 8, 0, 2),
        ],
    )
    def test_combined_representative(
        self,
        tws: float,
        twa: float,
        swh: float,
        mwa: float,
        v: float,
    ) -> None:
        """Hand-picked combined-condition cases."""
        ref = swopp3.predict_with_wps(tws, twa, swh, mwa, v)
        par = parametric_with_wps(tws, twa, swh, mwa, v)
        assert abs(par - ref) < 0.1, (
            f"tws={tws}, twa={twa}, swh={swh}, mwa={mwa}, v={v}: "
            f"ref={ref:.4f}, par={par:.4f}, err={abs(par - ref):.4f}"
        )

    @pytest.mark.parametrize(
        "tws",
        [0, 5, 10, 15, 20, 25, 30],
    )
    @pytest.mark.parametrize(
        "twa",
        [0, 45, 90, 135, 180],
    )
    @pytest.mark.parametrize(
        "v",
        [0, 4, 8, 12],
    )
    def test_grid_on_nodes(
        self,
        tws: float,
        twa: float,
        v: float,
    ) -> None:
        """Structured grid: tighter tolerance (closed-form, no interp)."""
        for swh, mwa in [(0, 0), (2, 45), (5, 120)]:
            ref = swopp3.predict_with_wps(tws, twa, swh, mwa, v)
            par = parametric_with_wps(tws, twa, swh, mwa, v)
            assert abs(par - ref) < 0.1, (
                f"tws={tws}, twa={twa}, swh={swh}, mwa={mwa}, v={v}: "
                f"ref={ref:.4f}, par={par:.4f}, err={abs(par - ref):.4f}"
            )

    def test_wps_always_leq_no_wps(self) -> None:
        """With WPS should never exceed without WPS (sails only help)."""
        rng = np.random.default_rng(123)
        for _ in range(500):
            tws = rng.uniform(0, 30)
            twa = rng.uniform(0, 180)
            swh = rng.uniform(0, 8)
            mwa = rng.uniform(0, 180)
            v = rng.uniform(0, 14.5)
            ref_no = swopp3.predict_no_wps(tws, twa, swh, mwa, v)
            ref_wp = swopp3.predict_with_wps(tws, twa, swh, mwa, v)
            par_no = parametric_no_wps(tws, twa, swh, mwa, v)
            par_wp = parametric_with_wps(tws, twa, swh, mwa, v)
            assert ref_wp <= ref_no + 1e-10
            assert par_wp <= par_no + 1e-10

    def test_random_stress_with_wps(self) -> None:
        """10 000 random inputs: fully closed-form, tight tolerance."""
        rng = np.random.default_rng(99)
        n = 10_000
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        ref_vals = np.array(
            [
                swopp3.predict_with_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
                for i in range(n)
            ]
        )
        par_vals = predict_power_batch(tws, twa, swh, mwa, v, wps=True)
        abs_errs = np.abs(par_vals - ref_vals)

        max_err = abs_errs.max()
        mean_err = abs_errs.mean()
        p99_err = np.percentile(abs_errs, 99)

        print(
            f"\n[with_wps random stress] n={n}, "
            f"max_err={max_err:.4f} kW, mean_err={mean_err:.4f} kW, "
            f"p99_err={p99_err:.4f} kW"
        )

        assert mean_err < 0.01, f"Mean absolute error too large: {mean_err:.4f} kW"
        assert p99_err < 0.1, f"p99 absolute error too large: {p99_err:.4f} kW"
        assert max_err < 0.15, f"Max absolute error too large: {max_err:.4f} kW"


# ===================================================================
#  Decomposition / additivity property tests
# ===================================================================
@needs_swopp3
class TestDecomposition:
    """Verify structural properties of the parametric decomposition."""

    def test_hull_cubic(self) -> None:
        """P_hull = K_h · v³ (pure cubic in speed, no env dependence)."""
        for v in [1, 3, 5, 8, 10, 12, 14]:
            # Hull-only = no wind, no waves
            ref = swopp3.predict_no_wps(0, 0, 0, 0, v)
            expected = K_H * v**3
            assert (
                abs(ref - expected) < 0.01
            ), f"v={v}: ref={ref:.4f}, K_h·v³={expected:.4f}"

    def test_additivity_wind_wave(self) -> None:
        """Wind and wave components are additive (no cross-terms)."""
        for tws, twa, swh, mwa, v in [
            (10, 45, 3, 60, 8),
            (15, 90, 2, 120, 5),
            (20, 0, 5, 0, 10),
        ]:
            p_hull = swopp3.predict_no_wps(0, 0, 0, 0, v)
            p_hull_wind = swopp3.predict_no_wps(tws, twa, 0, 0, v)
            p_hull_wave = swopp3.predict_no_wps(0, 0, swh, mwa, v)
            p_all = swopp3.predict_no_wps(tws, twa, swh, mwa, v)

            p_wind = p_hull_wind - p_hull
            p_wave = p_hull_wave - p_hull
            combined = p_hull + p_wind + p_wave

            # Only compare if not clamped
            if p_all > 0 and combined > 0:
                assert (
                    abs(p_all - combined) < 0.01
                ), f"Additivity fail: combined={combined:.4f}, ref={p_all:.4f}"

    def test_wave_swh_squared(self) -> None:
        """Wave power ∝ swh² at fixed (mwa, v)."""
        v = 8.0
        mwa = 45.0
        p1 = swopp3.predict_no_wps(0, 0, 1.0, mwa, v) - swopp3.predict_no_wps(
            0, 0, 0, 0, v
        )
        p2 = swopp3.predict_no_wps(0, 0, 2.0, mwa, v) - swopp3.predict_no_wps(
            0, 0, 0, 0, v
        )
        p4 = swopp3.predict_no_wps(0, 0, 4.0, mwa, v) - swopp3.predict_no_wps(
            0, 0, 0, 0, v
        )
        assert abs(p2 / p1 - 4.0) < 1e-6, f"ratio p2/p1 = {p2/p1:.6f}, expected 4"
        assert abs(p4 / p1 - 16.0) < 1e-6, f"ratio p4/p1 = {p4/p1:.6f}, expected 16"

    def test_wave_speed_exponent(self) -> None:
        """Wave power ∝ v^1.5 at fixed (swh, mwa)."""
        swh = 3.0
        mwa = 60.0
        powers = []
        speeds = [4.0, 6.0, 8.0, 10.0]
        for v in speeds:
            p = swopp3.predict_no_wps(0, 0, swh, mwa, v) - swopp3.predict_no_wps(
                0, 0, 0, 0, v
            )
            powers.append(p)

        for i in range(1, len(speeds)):
            ratio = powers[i] / powers[0]
            expected = (speeds[i] / speeds[0]) ** 1.5
            assert (
                abs(ratio - expected) < 1e-4
            ), f"v={speeds[i]}: ratio={ratio:.6f}, expected={expected:.6f}"

    def test_sail_wave_independent(self) -> None:
        """Sail power P_sail(tws,twa,v) is independent of waves.

        Uses high-speed scenarios with head-on waves (mwa=0) to keep
        total power well above zero, so the difference P_no − P_wps
        reveals the true sail thrust without hitting the clamp.
        All inputs stay within documented ranges (SWH ∈ [0, 10]).
        """
        for tws, twa, v in [(10, 90, 8), (20, 45, 10), (15, 135, 12)]:
            # Baseline sail power (moderate SWH, head-on waves)
            p_no_base = swopp3.predict_no_wps(tws, twa, 5.0, 0.0, v)
            p_wp_base = swopp3.predict_with_wps(tws, twa, 5.0, 0.0, v)
            assert p_wp_base > 0, "Baseline should not be clamped"
            p_sail_base = p_no_base - p_wp_base

            # Compare against different in-range wave conditions
            for swh, mwa in [(2.0, 0), (5.0, 45), (8.0, 0)]:
                p_no = swopp3.predict_no_wps(tws, twa, swh, mwa, v)
                p_wp = swopp3.predict_with_wps(tws, twa, swh, mwa, v)
                assert p_wp > 0, f"Should not be clamped: swh={swh}, mwa={mwa}"
                p_sail = p_no - p_wp
                assert abs(p_sail - p_sail_base) < 0.01, (
                    f"Sail depends on waves! mwa={mwa}: "
                    f"p_sail={p_sail:.4f} vs base={p_sail_base:.4f}"
                )


# ===================================================================
#  Public API tests (predict_power, predict_power_batch)
# ===================================================================
class TestPublicAPI:
    """Tests for the unified predict_power and batch entry points."""

    def test_predict_power_dispatches_no_wps(self) -> None:
        """predict_power(wps=False) matches predict_power_no_wps."""
        for tws, twa, swh, mwa, v in [(10, 90, 2, 45, 8), (0, 0, 0, 0, 5)]:
            expected = parametric_no_wps(tws, twa, swh, mwa, v)
            result = predict_power(tws, twa, swh, mwa, v, wps=False)
            assert result == expected

    def test_predict_power_dispatches_wps(self) -> None:
        """predict_power(wps=True) matches predict_power_with_wps."""
        for tws, twa, swh, mwa, v in [(10, 90, 2, 45, 8), (20, 135, 5, 150, 6)]:
            expected = parametric_with_wps(tws, twa, swh, mwa, v)
            result = predict_power(tws, twa, swh, mwa, v, wps=True)
            assert result == expected

    def test_predict_power_default_no_wps(self) -> None:
        """Default wps=False."""
        result = predict_power(10, 90, 2, 45, 8)
        expected = parametric_no_wps(10, 90, 2, 45, 8)
        assert result == expected

    def test_batch_matches_scalar_no_wps(self) -> None:
        """Batch output matches scalar loop for no-WPS mode."""
        rng = np.random.default_rng(777)
        n = 100
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        batch = predict_power_batch(tws, twa, swh, mwa, v, wps=False)
        for i in range(n):
            scalar = parametric_no_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
            assert (
                abs(batch[i] - scalar) < 1e-10
            ), f"i={i}: batch={batch[i]:.6f}, scalar={scalar:.6f}"

    def test_batch_matches_scalar_wps(self) -> None:
        """Batch output matches scalar loop for WPS mode."""
        rng = np.random.default_rng(888)
        n = 100
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        batch = predict_power_batch(tws, twa, swh, mwa, v, wps=True)
        for i in range(n):
            scalar = parametric_with_wps(tws[i], twa[i], swh[i], mwa[i], v[i])
            assert (
                abs(batch[i] - scalar) < 1e-10
            ), f"i={i}: batch={batch[i]:.6f}, scalar={scalar:.6f}"

    def test_batch_broadcasting(self) -> None:
        """Batch supports broadcasting (scalar v with array tws)."""
        tws = np.array([5, 10, 15, 20])
        result = predict_power_batch(tws, 90, 2, 45, 8, wps=False)
        assert result.shape == (4,)
        for i, tw in enumerate(tws):
            expected = parametric_no_wps(tw, 90, 2, 45, 8)
            assert abs(result[i] - expected) < 1e-10


class TestJaxParity:
    """Verify predict_power_jax matches predict_power_batch numerically."""

    def test_parity_no_wps(self) -> None:
        """JAX and NumPy outputs agree on a random grid (no WPS).

        JAX defaults to float32; tolerances reflect the difference
        vs NumPy float64.
        """
        from routetools.performance import predict_power_jax

        rng = np.random.default_rng(2025)
        n = 500
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        np_result = predict_power_batch(tws, twa, swh, mwa, v, wps=False)
        jax_result = np.asarray(
            predict_power_jax(
                jnp.array(tws),
                jnp.array(twa),
                jnp.array(swh),
                jnp.array(mwa),
                jnp.array(v),
                wps=False,
            )
        )

        np.testing.assert_allclose(
            jax_result,
            np_result,
            rtol=1e-5,
            atol=0.02,
            err_msg="predict_power_jax (no WPS) diverges from predict_power_batch",
        )

    def test_parity_with_wps(self) -> None:
        """JAX and NumPy outputs agree on a random grid (with WPS).

        JAX defaults to float32; tolerances reflect the difference
        vs NumPy float64.
        """
        from routetools.performance import predict_power_jax

        rng = np.random.default_rng(2026)
        n = 500
        tws = rng.uniform(0, 30, n)
        twa = rng.uniform(0, 180, n)
        swh = rng.uniform(0, 10, n)
        mwa = rng.uniform(0, 180, n)
        v = rng.uniform(0, 14.5, n)

        np_result = predict_power_batch(tws, twa, swh, mwa, v, wps=True)
        jax_result = np.asarray(
            predict_power_jax(
                jnp.array(tws),
                jnp.array(twa),
                jnp.array(swh),
                jnp.array(mwa),
                jnp.array(v),
                wps=True,
            )
        )

        np.testing.assert_allclose(
            jax_result,
            np_result,
            rtol=1e-5,
            atol=0.02,
            err_msg="predict_power_jax (WPS) diverges from predict_power_batch",
        )
