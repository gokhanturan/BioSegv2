"""richness_core modülü testleri: belirlenen tür sayısı ve seyreltme (elle hesaplanmış + sayım ile doğrulama)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.data import richness_core as re_

# Elle hesaplanan örnek (4 birim x 4 tür):
#   u1:{A,C,D}  u2:{B,C,D}  u3:{D}  u4:{D}
HAND = np.array([
    [1, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
], dtype=bool)


class TestHandComputed:
    def test_basic_counts(self):
        st = re_.incidence_stats(HAND)
        assert (st.m, st.s_obs) == (4, 4)
        assert list(st.freq) == [1, 1, 2, 4]

    def test_rarefaction_exact_by_enumeration(self):
        # Tüm 2'li birim kombinasyonları: 4,3,3,3,3,1 tür -> ort 17/6
        vals = [int(HAND[list(c)].any(axis=0).sum()) for c in itertools.combinations(range(4), 2)]
        mean, sd = re_.rarefaction(HAND, 2)
        assert mean == pytest.approx(np.mean(vals))
        assert sd == pytest.approx(np.std(vals))   # popülasyon s.s. (tüm alt kümeler)

    def test_rarefaction_endpoints(self):
        mean1, _ = re_.rarefaction(HAND, 1)
        assert mean1 == pytest.approx(HAND.sum() / 4)       # birim başına ortalama tür
        meanm, sdm = re_.rarefaction(HAND, 4)
        assert meanm == pytest.approx(4) and sdm == pytest.approx(0, abs=1e-9)


class TestRandomizedChecks:
    def test_rarefaction_enumeration_random_matrices(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            m, s = int(rng.integers(3, 9)), int(rng.integers(2, 12))
            arr = rng.random((m, s)) < rng.uniform(0.1, 0.7)
            arr = arr[:, arr.any(axis=0)]
            if arr.shape[1] == 0:
                continue
            for k in range(1, m + 1):
                vals = [int(arr[list(c)].any(axis=0).sum())
                        for c in itertools.combinations(range(m), k)]
                mean, sd = re_.rarefaction(arr, k)
                assert mean == pytest.approx(np.mean(vals), abs=1e-9)
                assert sd == pytest.approx(np.std(vals), abs=1e-7)


class TestEdgeCases:
    def test_empty_species_columns_are_dropped(self):
        arr = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        assert re_.incidence_stats(arr).s_obs == 1

    def test_rarefaction_k_out_of_range(self):
        with pytest.raises(ValueError):
            re_.rarefaction(HAND, 0)
        with pytest.raises(ValueError):
            re_.rarefaction(HAND, 5)

    def test_curve_shape(self):
        c = re_.rarefaction_curve(HAND)
        assert list(c["k"]) == [1, 2, 3, 4]
        assert c["beklenen_tur"].is_monotonic_increasing



class TestGroupRarefactionAndCurves:
    def _df(self):
        rows = []
        # A: 4 nokta, B: 2 nokta
        data = {"A": {"1": ["x", "y"], "2": ["x"], "3": ["z"], "4": ["y", "w"]},
                "B": {"5": ["x", "y"], "6": ["q"]}}
        for loc, pts in data.items():
            for pid, sps in pts.items():
                for sp in sps:
                    rows.append({"Location": loc, "Habitat_ID": pid, "Turkish_Name": sp})
        return pd.DataFrame(rows)

    def test_rarefy_groups_uses_min_unit_count(self):
        t = re_.rarefy_groups(self._df(), "Location")
        assert set(t["k"]) == {2}
        b = t[t["grup"] == "B"].iloc[0]
        assert b["seyreltilmis_tur"] == pytest.approx(b["s_obs"])   # k = m -> sayım
        assert b["ss"] == pytest.approx(0, abs=1e-9)
        a = t[t["grup"] == "A"].iloc[0]
        assert 2 <= a["seyreltilmis_tur"] <= a["s_obs"]


def test_new_richness_figures_render():
    import numpy as np
    import pandas as pd
    from src.visualization import richness_plots as rp
    rng = np.random.default_rng(0)
    inc = pd.DataFrame(rng.random((12, 20)) < 0.3, index=[str(i) for i in range(12)],
                       columns=[f"t{j}" for j in range(20)])
    types = pd.Series({str(i): ("A" if i % 2 else "B") for i in range(12)})
    eff = pd.DataFrame({"segment": rng.integers(2, 60, 12), "kayit": 1, "tur": inc.sum(axis=1).values},
                       index=inc.index)
    assert rp.species_incidence_histogram(inc).axes
    fig = rp.presence_heatmap(inc, types)
    assert fig.axes
    assert len(fig.axes[-1].get_yticklabels()) == int(inc.any(axis=0).sum())   # tüm türler
    assert rp.presence_heatmap(pd.DataFrame()).axes
    cm, named, other = rp.type_colors(["a"] * 3 + ["b"] * 2 + list("cdefghi"))
    assert len(named) == 9 and not other and len(set(cm.values())) == 9          # 9 tip: hepsi ayrı renk
    cm, named, other = rp.type_colors(list("abcdefghijkl"))
    assert len(named) == 10 and other and cm["l"] == "#B8B8B8" and len(set(list(cm.values())[:10])) == 10
