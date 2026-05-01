"""Tests for garuda.core.source_network.

Covers SourceNode, Separator, Reinjector, SourceGroup, and SourceNetwork.
"""

import numpy as np
import pytest

from garuda.core.source_network import (
    Reinjector,
    Separator,
    SourceGroup,
    SourceNetwork,
    SourceNode,
)


# ---------------------------------------------------------------------------
# SourceNode
# ---------------------------------------------------------------------------


class TestSourceNode:
    def test_default_construction(self):
        node = SourceNode(name="foo", cell_index=0)
        assert node.name == "foo"
        assert node.cell_index == 0
        assert node.rate == 0.0
        assert node.phase == "total"
        assert node.enthalpy is None
        assert node.active is True

    def test_explicit_construction(self):
        node = SourceNode(
            name="bar",
            cell_index=5,
            rate=-12.0,
            phase="water",
            enthalpy=2.5e6,
            active=False,
        )
        assert node.cell_index == 5
        assert node.rate == -12.0
        assert node.phase == "water"
        assert node.enthalpy == 2.5e6
        assert node.active is False


# ---------------------------------------------------------------------------
# Separator
# ---------------------------------------------------------------------------


class TestSeparator:
    def test_pass_through(self):
        sep = Separator("sep1")
        out = sep.separate(10.0)
        assert out == {"total": 10.0}
        assert sep.incoming_rate == 10.0

    def test_split_proportional(self):
        sep = Separator("sep2", {"water": 0.8, "steam": 0.2})
        out = sep.separate(100.0)
        assert pytest.approx(out["water"], 0.01) == 80.0
        assert pytest.approx(out["steam"], 0.01) == 20.0

    def test_zero_or_negative_input(self):
        sep = Separator("sep3", {"water": 0.5, "steam": 0.5})
        for val in [-5.0, 0.0]:
            out = sep.separate(val)
            assert out == {"total": val}

    def test_get_stream(self):
        sep = Separator("sep4", {"water": 1.0})
        sep.separate(42.0)
        assert sep.get_stream("water") == 42.0
        assert sep.get_stream("steam") == 0.0

    def test_get_stream_before_separate(self):
        sep = Separator("sep5", {"water": 1.0})
        assert sep.get_stream("water") == 0.0


# ---------------------------------------------------------------------------
# Reinjector
# ---------------------------------------------------------------------------


class TestReinjector:
    def test_construction(self):
        r = Reinjector("r1", cell_index=3, target_rate=10.0)
        assert r.name == "r1"
        assert r.cell_index == 3
        assert r.target_rate == 10.0
        assert r.active is True
        assert r.current_rate == 0.0

    def test_compute_rate_when_active(self):
        r = Reinjector("r2", cell_index=0, target_rate=5.0)
        assert r.compute_rate(available_mass=3.0) == 3.0
        assert r.current_rate == 3.0
        assert r.compute_rate(available_mass=10.0) == 5.0
        assert r.current_rate == 5.0

    def test_compute_rate_when_inactive(self):
        r = Reinjector("r3", cell_index=0, target_rate=5.0)
        r.active = False
        assert r.compute_rate(available_mass=100.0) == 0.0
        assert r.current_rate == 0.0

    def test_compute_rate_no_available_mass(self):
        r = Reinjector("r4", cell_index=0, target_rate=5.0)
        assert r.compute_rate(available_mass=0.0) == 0.0
        assert r.compute_rate(available_mass=-1.0) == 0.0

    def test_injection_enthalpy(self):
        r = Reinjector("r5", cell_index=0, injection_temperature=298.15)
        h = r.injection_enthalpy()
        assert h > 0.0
        assert pytest.approx(h, rel=1e-3) == 4182.0 * 25.0


# ---------------------------------------------------------------------------
# SourceGroup
# ---------------------------------------------------------------------------


class TestSourceGroup:
    def test_add_and_remove_node(self):
        g = SourceGroup("g1")
        n = SourceNode("n1", cell_index=0, rate=10.0)
        g.add_node(n)
        assert len(g.nodes) == 1
        assert g.get_node("n1") is n
        g.remove_node("n1")
        assert g.get_node("n1") is None

    def test_compute_group_rate(self):
        g = SourceGroup("g2")
        g.add_node(SourceNode("a", cell_index=0, rate=10.0))
        g.add_node(SourceNode("b", cell_index=1, rate=-3.0))
        assert g.compute_group_rate() == 7.0

    def test_compute_group_rate_ignores_inactive(self):
        g = SourceGroup("g3")
        n = SourceNode("a", cell_index=0, rate=10.0, active=False)
        g.add_node(n)
        assert g.compute_group_rate() == 0.0

    def test_allocate_uniform(self):
        g = SourceGroup("g4")
        g.add_node(SourceNode("a", cell_index=0, rate=1.0))
        g.add_node(SourceNode("b", cell_index=1, rate=2.0))
        g.allocate_rates(10.0, method="uniform")
        assert g.nodes["a"].rate == 5.0
        assert g.nodes["b"].rate == 5.0

    def test_allocate_proportional(self):
        g = SourceGroup("g5")
        g.add_node(SourceNode("a", cell_index=0, rate=2.0))
        g.add_node(SourceNode("b", cell_index=1, rate=8.0))
        g.allocate_rates(5.0, method="proportional")
        assert pytest.approx(g.nodes["a"].rate, abs=1e-6) == 1.0
        assert pytest.approx(g.nodes["b"].rate, abs=1e-6) == 4.0

    def test_allocate_proportional_zero_sum(self):
        g = SourceGroup("g6")
        g.add_node(SourceNode("a", cell_index=0, rate=0.0))
        g.add_node(SourceNode("b", cell_index=1, rate=0.0))
        g.allocate_rates(10.0, method="proportional")
        assert g.nodes["a"].rate == 5.0
        assert g.nodes["b"].rate == 5.0

    def test_allocate_bad_method(self):
        g = SourceGroup("g7")
        with pytest.raises(ValueError):
            g.allocate_rates(1.0, method="magic")

    def test_get_source_terms(self):
        g = SourceGroup("g8")
        g.add_node(SourceNode("a", cell_index=2, rate=5.0))
        g.add_node(SourceNode("b", cell_index=2, rate=-1.0))
        g.add_node(SourceNode("c", cell_index=99, rate=3.0))  # out of range
        terms = g.get_source_terms(num_cells=10)
        assert len(terms) == 10
        assert terms[2] == 4.0

    def test_get_enthalpy_terms(self):
        g = SourceGroup("g9")
        g.add_node(SourceNode("a", cell_index=1, rate=2.0, enthalpy=100.0))
        h = g.get_enthalpy_terms(num_cells=5)
        assert h[1] == 200.0
        assert h[0] == 0.0


# ---------------------------------------------------------------------------
# SourceNetwork
# ---------------------------------------------------------------------------


class TestSourceNetwork:
    def test_empty_network(self):
        net = SourceNetwork()
        terms = net.compute_source_terms(num_cells=5)
        assert terms["mass"].shape == (5,)
        assert terms["enthalpy"].shape == (5,)
        assert np.all(terms["mass"] == 0.0)
        assert np.all(terms["enthalpy"] == 0.0)

    def test_add_and_get(self):
        net = SourceNetwork()
        g = SourceGroup("gg")
        net.add_group(g)
        assert net.get_group("gg") is g
        assert net.get_group("missing") is None

    def test_connect_invalid_types(self):
        net = SourceNetwork()
        with pytest.raises(ValueError):
            net.connect("bad", "x", "separator", "y")
        with pytest.raises(ValueError):
            net.connect("group", "x", "bad", "y")

    def test_produce_separator_reinject(self):
        # Full chain: producer -> separator -> reinjector
        net = SourceNetwork()

        prod = SourceNode("prod", cell_index=0, rate=-50.0, enthalpy=2.6e6)
        group = SourceGroup("wells")
        group.add_node(prod)

        sep = Separator("sep", {"water": 0.9, "steam": 0.1})
        reinj = Reinjector("reinj", cell_index=1, target_rate=45.0)

        net.add_group(group)
        net.add_separator(sep)
        net.add_reinjector(reinj)
        net.connect("group", "wells", "separator", "sep")
        net.connect("separator", "sep", "reinjector", "reinj", "water")

        terms = net.compute_source_terms(num_cells=5)
        assert terms["mass"][0] == -50.0
        assert terms["mass"][1] == pytest.approx(45.0, abs=1e-6)
        assert terms["enthalpy"][0] == pytest.approx(-50.0 * 2.6e6, rel=1e-6)
        assert terms["enthalpy"][1] > 0.0

    def test_remove_connection(self):
        net = SourceNetwork()
        g = SourceGroup("g")
        sep = Separator("s")
        net.add_group(g)
        net.add_separator(sep)
        net.connect("group", "g", "separator", "s")
        assert len(net.connections) == 1
        net.remove_connection("g", "s")
        assert len(net.connections) == 0

    def test_repr(self):
        net = SourceNetwork()
        net.add_group(SourceGroup("g1"))
        r = repr(net)
        assert "groups=1" in r
        assert "SourceNetwork" in r

    def test_multiple_producers_into_separator(self):
        net = SourceNetwork()
        g1 = SourceGroup("g1")
        g1.add_node(SourceNode("n1", 0, rate=-10.0))
        g2 = SourceGroup("g2")
        g2.add_node(SourceNode("n2", 1, rate=-20.0))
        sep = Separator("sep", {"water": 1.0})
        reinj = Reinjector("r", cell_index=2, target_rate=100.0)

        net.add_group(g1)
        net.add_group(g2)
        net.add_separator(sep)
        net.add_reinjector(reinj)
        net.connect("group", "g1", "separator", "sep")
        net.connect("group", "g2", "separator", "sep")
        net.connect("separator", "sep", "reinjector", "r", "water")

        terms = net.compute_source_terms(num_cells=5)
        # Total prod = -30, separated 100% water, reinj limited by available = 30.
        assert terms["mass"][0] == -10.0
        assert terms["mass"][1] == -20.0
        assert terms["mass"][2] == pytest.approx(30.0, abs=1e-6)
