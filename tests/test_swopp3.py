"""Tests for routetools.swopp3 — SWOPP3 configuration and helpers."""

from datetime import datetime, timezone

import jax.numpy as jnp
import pytest

from routetools.swopp3 import (
    PORTS,
    ROUTE_ATLANTIC,
    ROUTE_PACIFIC,
    SWOPP3_CASES,
    case_endpoints,
    case_travel_time_seconds,
    departure_strings,
    departures_2024,
    great_circle_route,
)


# ---------------------------------------------------------------------------
# Port definitions
# ---------------------------------------------------------------------------
class TestPorts:
    def test_all_four_ports_defined(self):
        assert set(PORTS.keys()) == {"ESSDR", "USNYC", "JPTYO", "USLAX"}

    def test_santander_coords(self):
        assert PORTS["ESSDR"]["lat"] == 43.6
        assert PORTS["ESSDR"]["lon"] == -4.0

    def test_new_york_coords(self):
        assert PORTS["USNYC"]["lat"] == 40.53
        assert PORTS["USNYC"]["lon"] == -73.80

    def test_tokyo_coords(self):
        assert PORTS["JPTYO"]["lat"] == 34.8
        assert PORTS["JPTYO"]["lon"] == 140.0

    def test_la_coords(self):
        assert PORTS["USLAX"]["lat"] == 34.4
        assert PORTS["USLAX"]["lon"] == -121.0


# ---------------------------------------------------------------------------
# Route definitions
# ---------------------------------------------------------------------------
class TestRoutes:
    def test_atlantic_passage(self):
        assert ROUTE_ATLANTIC["passage_hours"] == 354

    def test_pacific_passage(self):
        assert ROUTE_PACIFIC["passage_hours"] == 583

    def test_atlantic_ports(self):
        assert ROUTE_ATLANTIC["src_port"] == "ESSDR"
        assert ROUTE_ATLANTIC["dst_port"] == "USNYC"

    def test_pacific_ports(self):
        assert ROUTE_PACIFIC["src_port"] == "JPTYO"
        assert ROUTE_PACIFIC["dst_port"] == "USLAX"


# ---------------------------------------------------------------------------
# 8 SWOPP3 cases
# ---------------------------------------------------------------------------
_CASE_IDS = ["AO_WPS", "AO_noWPS", "AGC_WPS", "AGC_noWPS",
             "PO_WPS", "PO_noWPS", "PGC_WPS", "PGC_noWPS"]

_ATLANTIC_CASES = ["AO_WPS", "AO_noWPS", "AGC_WPS", "AGC_noWPS"]
_PACIFIC_CASES = ["PO_WPS", "PO_noWPS", "PGC_WPS", "PGC_noWPS"]


class TestCases:
    def test_eight_cases(self):
        assert len(SWOPP3_CASES) == 8

    def test_case_ids(self):
        assert set(SWOPP3_CASES.keys()) == set(_CASE_IDS)

    @pytest.mark.parametrize("cid", _CASE_IDS)
    def test_case_has_required_keys(self, cid):
        case = SWOPP3_CASES[cid]
        for key in ("name", "label", "src_port", "dst_port",
                     "passage_hours", "route", "strategy", "wps"):
            assert key in case, f"{cid} missing key {key}"

    @pytest.mark.parametrize("cid", _CASE_IDS)
    def test_strategy_values(self, cid):
        assert SWOPP3_CASES[cid]["strategy"] in ("optimised", "gc")

    def test_optimised_cases(self):
        for cid in ("AO_WPS", "AO_noWPS", "PO_WPS", "PO_noWPS"):
            assert SWOPP3_CASES[cid]["strategy"] == "optimised"

    def test_gc_cases(self):
        for cid in ("AGC_WPS", "AGC_noWPS", "PGC_WPS", "PGC_noWPS"):
            assert SWOPP3_CASES[cid]["strategy"] == "gc"

    def test_wps_flag(self):
        for cid in ("AO_WPS", "AGC_WPS", "PO_WPS", "PGC_WPS"):
            assert SWOPP3_CASES[cid]["wps"] is True
        for cid in ("AO_noWPS", "AGC_noWPS", "PO_noWPS", "PGC_noWPS"):
            assert SWOPP3_CASES[cid]["wps"] is False

    def test_atlantic_passage_hours(self):
        for cid in _ATLANTIC_CASES:
            assert SWOPP3_CASES[cid]["passage_hours"] == 354

    def test_pacific_passage_hours(self):
        for cid in _PACIFIC_CASES:
            assert SWOPP3_CASES[cid]["passage_hours"] == 583

    def test_atlantic_route(self):
        for cid in _ATLANTIC_CASES:
            assert SWOPP3_CASES[cid]["src_port"] == "ESSDR"
            assert SWOPP3_CASES[cid]["dst_port"] == "USNYC"

    def test_pacific_route(self):
        for cid in _PACIFIC_CASES:
            assert SWOPP3_CASES[cid]["src_port"] == "JPTYO"
            assert SWOPP3_CASES[cid]["dst_port"] == "USLAX"


# ---------------------------------------------------------------------------
# Departures
# ---------------------------------------------------------------------------
class TestDepartures:
    def test_366_departures(self):
        deps = departures_2024()
        assert len(deps) == 366

    def test_first_departure(self):
        deps = departures_2024()
        expected = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert deps[0] == expected

    def test_last_departure(self):
        deps = departures_2024()
        expected = datetime(2024, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
        assert deps[-1] == expected

    def test_all_noon_utc(self):
        for d in departures_2024():
            assert d.hour == 12
            assert d.minute == 0
            assert d.second == 0

    def test_consecutive_days(self):
        deps = departures_2024()
        for i in range(1, len(deps)):
            delta = deps[i] - deps[i - 1]
            assert delta.days == 1
            assert delta.seconds == 0

    def test_departure_strings_iso(self):
        strings = departure_strings()
        assert len(strings) == 366
        assert strings[0] == "2024-01-01T12:00:00"
        assert strings[-1] == "2024-12-31T12:00:00"

    def test_departure_strings_custom_format(self):
        strings = departure_strings(fmt="%Y-%m-%d")
        assert strings[0] == "2024-01-01"


# ---------------------------------------------------------------------------
# case_endpoints
# ---------------------------------------------------------------------------
class TestCaseEndpoints:
    def test_ao_wps_src_dst(self):
        src, dst = case_endpoints("AO_WPS")
        assert src.shape == (2,)
        assert dst.shape == (2,)
        # src should be Santander (lon, lat)
        assert jnp.allclose(src, jnp.array([-4.0, 43.6]))
        # dst should be New York (lon, lat)
        assert jnp.allclose(dst, jnp.array([-73.80, 40.53]))

    def test_all_atlantic_same_endpoints(self):
        src_ref, dst_ref = case_endpoints("AO_WPS")
        for cid in _ATLANTIC_CASES:
            src, dst = case_endpoints(cid)
            assert jnp.allclose(src, src_ref)
            assert jnp.allclose(dst, dst_ref)

    def test_po_wps_pacific(self):
        src, dst = case_endpoints("PO_WPS")
        # Tokyo
        assert jnp.allclose(src, jnp.array([140.0, 34.8]))
        # LA
        assert jnp.allclose(dst, jnp.array([-121.0, 34.4]))

    def test_all_pacific_same_endpoints(self):
        src_ref, dst_ref = case_endpoints("PO_WPS")
        for cid in _PACIFIC_CASES:
            src, dst = case_endpoints(cid)
            assert jnp.allclose(src, src_ref)
            assert jnp.allclose(dst, dst_ref)

    def test_invalid_case_raises(self):
        with pytest.raises(KeyError):
            case_endpoints("case99")


# ---------------------------------------------------------------------------
# case_travel_time_seconds
# ---------------------------------------------------------------------------
class TestCaseTravelTime:
    def test_atlantic(self):
        assert case_travel_time_seconds("AO_WPS") == 354 * 3600.0

    def test_pacific(self):
        assert case_travel_time_seconds("PO_WPS") == 583 * 3600.0


# ---------------------------------------------------------------------------
# Great-circle route
# ---------------------------------------------------------------------------
class TestGreatCircle:
    def test_endpoints_match(self):
        src = jnp.array([-4.0, 43.6])
        dst = jnp.array([-73.80, 40.53])
        route = great_circle_route(src, dst, n_points=50)
        assert route.shape == (50, 2)
        assert jnp.allclose(route[0], src, atol=1e-4)
        assert jnp.allclose(route[-1], dst, atol=1e-4)

    def test_pacific_antimeridian(self):
        """Pacific route crosses the antimeridian — no wrapping artifact."""
        src = jnp.array([140.0, 34.8])  # Tokyo
        dst = jnp.array([-121.0, 34.4])  # LA
        route = great_circle_route(src, dst, n_points=200)
        # Route should go eastward across the Pacific (longitudes > 140 or < -121)
        # The key test: no sudden 360° jumps between consecutive points
        dlon = jnp.diff(route[:, 0])
        assert jnp.all(jnp.abs(dlon) < 10.0), "Large longitude jump detected — antimeridian bug"

    def test_coincident_points(self):
        """Degenerate case: src == dst."""
        pt = jnp.array([10.0, 50.0])
        route = great_circle_route(pt, pt, n_points=10)
        assert route.shape == (10, 2)
        # All points should be at the same location
        for i in range(10):
            assert jnp.allclose(route[i], pt, atol=1e-4)

    def test_route_latitude_range(self):
        """Atlantic GC route should curve northward (higher than both endpoints)."""
        src = jnp.array([-4.0, 43.6])  # Santander
        dst = jnp.array([-73.80, 40.53])  # NYC
        route = great_circle_route(src, dst, n_points=100)
        max_lat = jnp.max(route[:, 1])
        # GC route from Santander to NYC curves north — max latitude > both endpoints
        assert max_lat > max(43.6, 40.53), f"max_lat {max_lat} should exceed endpoint lats"
