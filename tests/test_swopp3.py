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
class TestCases:
    def test_eight_cases(self):
        assert len(SWOPP3_CASES) == 8

    def test_case_ids(self):
        expected = {f"case{i}" for i in range(1, 9)}
        assert set(SWOPP3_CASES.keys()) == expected

    @pytest.mark.parametrize("cid", [f"case{i}" for i in range(1, 9)])
    def test_case_has_required_keys(self, cid):
        case = SWOPP3_CASES[cid]
        for key in ("name", "label", "src_port", "dst_port", "passage_hours", "weather_constrained"):
            assert key in case, f"{cid} missing key {key}"

    def test_atlantic_cases_weather(self):
        assert SWOPP3_CASES["case1"]["weather_constrained"] is True
        assert SWOPP3_CASES["case2"]["weather_constrained"] is True
        assert SWOPP3_CASES["case3"]["weather_constrained"] is False
        assert SWOPP3_CASES["case4"]["weather_constrained"] is False

    def test_pacific_cases_weather(self):
        assert SWOPP3_CASES["case5"]["weather_constrained"] is True
        assert SWOPP3_CASES["case6"]["weather_constrained"] is True
        assert SWOPP3_CASES["case7"]["weather_constrained"] is False
        assert SWOPP3_CASES["case8"]["weather_constrained"] is False

    def test_atlantic_passage_hours(self):
        for cid in ("case1", "case2", "case3", "case4"):
            assert SWOPP3_CASES[cid]["passage_hours"] == 354

    def test_pacific_passage_hours(self):
        for cid in ("case5", "case6", "case7", "case8"):
            assert SWOPP3_CASES[cid]["passage_hours"] == 583

    def test_reverse_routes_paired(self):
        # case1/case2 are reverse of each other
        assert SWOPP3_CASES["case1"]["src_port"] == SWOPP3_CASES["case2"]["dst_port"]
        assert SWOPP3_CASES["case1"]["dst_port"] == SWOPP3_CASES["case2"]["src_port"]
        # case5/case6 are reverse of each other
        assert SWOPP3_CASES["case5"]["src_port"] == SWOPP3_CASES["case6"]["dst_port"]
        assert SWOPP3_CASES["case5"]["dst_port"] == SWOPP3_CASES["case6"]["src_port"]


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
    def test_case1_src_dst(self):
        src, dst = case_endpoints("case1")
        assert src.shape == (2,)
        assert dst.shape == (2,)
        # src should be Santander (lon, lat)
        assert jnp.allclose(src, jnp.array([-4.0, 43.6]))
        # dst should be New York (lon, lat)
        assert jnp.allclose(dst, jnp.array([-73.80, 40.53]))

    def test_case2_is_reverse_of_case1(self):
        src1, dst1 = case_endpoints("case1")
        src2, dst2 = case_endpoints("case2")
        assert jnp.allclose(src1, dst2)
        assert jnp.allclose(dst1, src2)

    def test_case5_pacific(self):
        src, dst = case_endpoints("case5")
        # Tokyo
        assert jnp.allclose(src, jnp.array([140.0, 34.8]))
        # LA
        assert jnp.allclose(dst, jnp.array([-121.0, 34.4]))

    def test_invalid_case_raises(self):
        with pytest.raises(KeyError):
            case_endpoints("case99")


# ---------------------------------------------------------------------------
# case_travel_time_seconds
# ---------------------------------------------------------------------------
class TestCaseTravelTime:
    def test_atlantic(self):
        assert case_travel_time_seconds("case1") == 354 * 3600.0

    def test_pacific(self):
        assert case_travel_time_seconds("case5") == 583 * 3600.0


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
