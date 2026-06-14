"""Tests déterministes pour le module fingerprint (v0.48).

Organisation :
- Tests 1–19  : canonicalize_for_fingerprint / fingerprint_payload
- Tests 20–33 : fingerprint_dataframe
- Tests 34–42 : build_input_fingerprints
"""
from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_agent.fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    build_input_fingerprints,
    canonicalize_for_fingerprint,
    fingerprint_dataframe,
    fingerprint_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256(value: str) -> bool:
    return bool(SHA256_RE.match(value))


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
    })


# ---------------------------------------------------------------------------
# 1. Même mapping → même hash
# ---------------------------------------------------------------------------

def test_same_mapping_same_hash():
    assert fingerprint_payload({"a": 1, "b": 2}) == fingerprint_payload({"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# 2. Ordre de clés différent → même hash
# ---------------------------------------------------------------------------

def test_different_key_order_same_hash():
    assert fingerprint_payload({"a": 1, "b": 2}) == fingerprint_payload({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# 3. Mapping imbriqué, ordre de clés différent → même hash
# ---------------------------------------------------------------------------

def test_nested_mapping_key_order_same_hash():
    a = {"x": {"a": 1, "b": 2}}
    b = {"x": {"b": 2, "a": 1}}
    assert fingerprint_payload(a) == fingerprint_payload(b)


# ---------------------------------------------------------------------------
# 4. Valeur différente → hash différent
# ---------------------------------------------------------------------------

def test_different_value_different_hash():
    assert fingerprint_payload({"a": 1}) != fingerprint_payload({"a": 2})


# ---------------------------------------------------------------------------
# 5. Ordre d'une liste différent → hash différent
# ---------------------------------------------------------------------------

def test_list_order_matters():
    assert fingerprint_payload({"a": [1, 2]}) != fingerprint_payload({"a": [2, 1]})


# ---------------------------------------------------------------------------
# 6. Tuple et liste → traités de façon identique (documenté)
# ---------------------------------------------------------------------------

def test_tuple_and_list_same_hash():
    # tuples sont canonicalisés comme des listes : comportement intentionnel et documenté
    assert fingerprint_payload([1, 2, 3]) == fingerprint_payload((1, 2, 3))


# ---------------------------------------------------------------------------
# 7. Path sérialisé comme une chaîne
# ---------------------------------------------------------------------------

def test_path_serialized_as_string():
    assert fingerprint_payload(Path("/tmp/foo")) == fingerprint_payload("/tmp/foo")


# ---------------------------------------------------------------------------
# 8. Scalaires NumPy pris en charge
# ---------------------------------------------------------------------------

def test_numpy_scalar_supported():
    assert fingerprint_payload({"v": np.int64(42)}) == fingerprint_payload({"v": 42})
    assert fingerprint_payload(np.float64(1.5)) == fingerprint_payload(1.5)
    assert fingerprint_payload(np.bool_(True)) == fingerprint_payload(True)


# ---------------------------------------------------------------------------
# 9. Timestamp naïf stable
# ---------------------------------------------------------------------------

def test_naive_timestamp_stable():
    ts = pd.Timestamp("2024-01-15T12:00:00")
    c1 = canonicalize_for_fingerprint(ts)
    c2 = canonicalize_for_fingerprint(ts)
    assert c1 == c2
    assert isinstance(c1, str)


# ---------------------------------------------------------------------------
# 10. Timestamps timezone-aware équivalents → même canonique UTC
# ---------------------------------------------------------------------------

def test_timezone_aware_equivalent_instant():
    ts_utc = pd.Timestamp("2024-01-15T10:00:00+00:00")
    ts_plus2 = pd.Timestamp("2024-01-15T12:00:00+02:00")
    assert canonicalize_for_fingerprint(ts_utc) == canonicalize_for_fingerprint(ts_plus2)


# ---------------------------------------------------------------------------
# 11. NaN stable
# ---------------------------------------------------------------------------

def test_nan_stable():
    c1 = canonicalize_for_fingerprint(float("nan"))
    c2 = canonicalize_for_fingerprint(float("nan"))
    assert c1 == c2
    assert c1 == {"__special__": "nan"}


# ---------------------------------------------------------------------------
# 12. +inf et -inf distincts
# ---------------------------------------------------------------------------

def test_inf_distinct():
    pos = canonicalize_for_fingerprint(float("+inf"))
    neg = canonicalize_for_fingerprint(float("-inf"))
    assert pos != neg
    assert pos == {"__special__": "+inf"}
    assert neg == {"__special__": "-inf"}


# ---------------------------------------------------------------------------
# 13. set refusé
# ---------------------------------------------------------------------------

def test_set_rejected():
    with pytest.raises(TypeError, match="set"):
        canonicalize_for_fingerprint({1, 2, 3})


# ---------------------------------------------------------------------------
# 14. Objet inconnu refusé
# ---------------------------------------------------------------------------

def test_unknown_object_rejected():
    with pytest.raises(TypeError):
        canonicalize_for_fingerprint(object())


# ---------------------------------------------------------------------------
# 15. Argument original non modifié
# ---------------------------------------------------------------------------

def test_original_argument_not_modified():
    original = {"b": 2, "a": 1}
    original_copy = dict(original)
    original_keys_before = list(original.keys())
    fingerprint_payload(original)
    assert list(original.keys()) == original_keys_before
    assert original == original_copy


# ---------------------------------------------------------------------------
# 16. Résultat SHA-256 = 64 caractères hexadécimaux
# ---------------------------------------------------------------------------

def test_sha256_hex_64_chars():
    result = fingerprint_payload({"a": 1})
    assert _is_sha256(result), f"pas un SHA-256 valide : {result!r}"


# ---------------------------------------------------------------------------
# 17. Mapping imbriqué avec clés textuelles accepté (ajustement)
# ---------------------------------------------------------------------------

def test_mapping_with_string_keys_accepted():
    nested = OrderedDict([("b", OrderedDict([("y", 2), ("x", 1)])), ("a", 0)])
    # ne doit pas lever d'erreur
    result = fingerprint_payload(nested)
    assert _is_sha256(result)
    # insensible à l'ordre des clés
    expected = fingerprint_payload({"a": 0, "b": {"x": 1, "y": 2}})
    assert result == expected


# ---------------------------------------------------------------------------
# 18. Clé non textuelle refusée (ajustement)
# ---------------------------------------------------------------------------

def test_non_string_key_rejected():
    with pytest.raises(TypeError, match="str"):
        canonicalize_for_fingerprint({1: "a"})

    with pytest.raises(TypeError, match="str"):
        canonicalize_for_fingerprint({True: "a"})


# ---------------------------------------------------------------------------
# 19. Pas de collision silencieuse entre clé int et clé str (ajustement)
# ---------------------------------------------------------------------------

def test_no_silent_collision_int_vs_str_key():
    # {1: "a"} doit lever TypeError avant même de calculer un hash
    # donc aucune collision avec {"1": "a"} n'est possible
    with pytest.raises(TypeError):
        fingerprint_payload({1: "a"})
    # la version texte fonctionne normalement
    assert _is_sha256(fingerprint_payload({"1": "a"}))


# ---------------------------------------------------------------------------
# 19b. Mapping avec clés hétérogènes (int + str) — TypeError contrôlé avant tri
# ---------------------------------------------------------------------------

def test_heterogeneous_keys_raise_controlled_typeerror():
    # Python ne peut pas trier {1: ..., "1": ...} nativement (TypeError lors du tri).
    # La branche Mapping doit valider les clés AVANT le tri pour lever une erreur
    # explicite avec le message attendu, et non un TypeError natif opaque.
    mixed = {1: "integer-key", "1": "string-key"}
    original_repr = dict(mixed)  # copie pour vérifier l'immutabilité

    with pytest.raises(TypeError, match="mapping keys must be str"):
        canonicalize_for_fingerprint(mixed)

    # le mapping d'origine doit rester inchangé
    assert dict(mixed) == original_repr


# ---------------------------------------------------------------------------
# 20. Copie identique → même hash
# ---------------------------------------------------------------------------

def test_dataframe_copy_same_hash():
    df = _simple_df()
    assert fingerprint_dataframe(df) == fingerprint_dataframe(df.copy())


# ---------------------------------------------------------------------------
# 21. Ordre des colonnes différent → même hash
# ---------------------------------------------------------------------------

def test_dataframe_column_order_same_hash():
    df = _simple_df()
    df_reordered = df[["close", "volume", "open"]]
    assert fingerprint_dataframe(df) == fingerprint_dataframe(df_reordered)


# ---------------------------------------------------------------------------
# 22. Index différent → même hash
# ---------------------------------------------------------------------------

def test_dataframe_different_index_same_hash():
    df = _simple_df()
    df2 = df.copy()
    df2.index = [100, 200, 300]
    assert fingerprint_dataframe(df) == fingerprint_dataframe(df2)


# ---------------------------------------------------------------------------
# 23. Valeur modifiée → hash différent
# ---------------------------------------------------------------------------

def test_dataframe_different_value_different_hash():
    df = _simple_df()
    df2 = df.copy()
    df2.loc[0, "open"] = 99.0
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df2)


# ---------------------------------------------------------------------------
# 24. Ordre des lignes différent → hash différent
# ---------------------------------------------------------------------------

def test_dataframe_row_order_matters():
    df = _simple_df()
    df_shuffled = df.iloc[::-1].reset_index(drop=True)
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df_shuffled)


# ---------------------------------------------------------------------------
# 25. Dtype différent → hash différent
# ---------------------------------------------------------------------------

def test_dataframe_different_dtype_different_hash():
    df = _simple_df()
    df2 = df.copy()
    df2["volume"] = df2["volume"].astype("float64")
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df2)


# ---------------------------------------------------------------------------
# 26. Timestamps tz-aware équivalents avec même dtype (object) → même hash
#
# Note : deux DataFrames dont le dtype de colonne diffère produisent des hashes
# différents (car les dtypes font partie de l'empreinte — test 25). Ce test
# utilise des colonnes de dtype "object" pour isoler uniquement l'équivalence
# d'instant sans changer de dtype.
# ---------------------------------------------------------------------------

def test_dataframe_timestamp_tz_equivalent_deterministic():
    import datetime as dt
    ts_utc = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    ts_plus2 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone(dt.timedelta(hours=2)))
    # Forcer dtype=object pour que les deux colonnes aient le même dtype ;
    # pandas inférerait sinon des dtypes distincts (datetime64[ns, UTC] vs
    # datetime64[ns, pytz.FixedOffset(120)]), ce qui changerait le hash par
    # la règle "dtype différent → hash différent".
    df_utc = pd.DataFrame({"ts": pd.Series([ts_utc], dtype=object)})
    df_plus2 = pd.DataFrame({"ts": pd.Series([ts_plus2], dtype=object)})
    # Même instant, même dtype object, canonicalisé vers le même instant UTC
    assert fingerprint_dataframe(df_utc) == fingerprint_dataframe(df_plus2)


# ---------------------------------------------------------------------------
# 27. Valeur NaN → représentation stable
# ---------------------------------------------------------------------------

def test_dataframe_nan_stable():
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    assert fingerprint_dataframe(df) == fingerprint_dataframe(df.copy())


# ---------------------------------------------------------------------------
# 28. DataFrame vide → supporté sans erreur
# ---------------------------------------------------------------------------

def test_dataframe_empty_supported():
    df_empty = pd.DataFrame()
    result = fingerprint_dataframe(df_empty)
    assert _is_sha256(result)

    df_empty_cols = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    result2 = fingerprint_dataframe(df_empty_cols)
    assert _is_sha256(result2)


# ---------------------------------------------------------------------------
# 29. DataFrame original non modifié
# ---------------------------------------------------------------------------

def test_dataframe_original_not_modified():
    df = _simple_df()
    columns_before = df.columns.tolist()
    values_before = df.values.copy()
    index_before = df.index.tolist()
    fingerprint_dataframe(df)
    assert df.columns.tolist() == columns_before
    assert (df.values == values_before).all()
    assert df.index.tolist() == index_before


# ---------------------------------------------------------------------------
# 30. Colonne ajoutée → hash différent
# ---------------------------------------------------------------------------

def test_dataframe_added_column_different_hash():
    df = _simple_df()
    df2 = df.copy()
    df2["extra"] = 0.0
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df2)


# ---------------------------------------------------------------------------
# 31. Ligne supprimée → hash différent
# ---------------------------------------------------------------------------

def test_dataframe_removed_row_different_hash():
    df = _simple_df()
    df2 = df.iloc[:-1].reset_index(drop=True)
    assert fingerprint_dataframe(df) != fingerprint_dataframe(df2)


# ---------------------------------------------------------------------------
# 32. Noms de colonnes non textuels refusés (ajustement)
# ---------------------------------------------------------------------------

def test_dataframe_non_string_column_rejected():
    df = pd.DataFrame({1: [1.0], 2: [2.0]})
    with pytest.raises(TypeError, match="str"):
        fingerprint_dataframe(df)


# ---------------------------------------------------------------------------
# 33. Colonnes dupliquées refusées (ajustement)
# ---------------------------------------------------------------------------

def test_dataframe_duplicate_column_rejected():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        fingerprint_dataframe(df)


# ---------------------------------------------------------------------------
# 34. build_input_fingerprints — toutes les clés attendues
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "fingerprint_schema_version",
    "dataframe_sha256",
    "strategy_sha256",
    "goal_sha256",
    "validation_policy_sha256",
}


def test_build_all_keys_present():
    result = build_input_fingerprints(
        dataframe=_simple_df(),
        strategy={"name": "sma"},
        goal={"target": "exploratory"},
    )
    assert set(result.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# 35. fingerprint_schema_version == 1
# ---------------------------------------------------------------------------

def test_build_schema_version():
    result = build_input_fingerprints(
        dataframe=_simple_df(),
        strategy={"name": "sma"},
        goal={"target": "exploratory"},
    )
    assert result["fingerprint_schema_version"] == FINGERPRINT_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# 36. Même entrées → même résultat (déterminisme)
# ---------------------------------------------------------------------------

def test_build_deterministic():
    kwargs = dict(
        dataframe=_simple_df(),
        strategy={"fast": 10, "slow": 30},
        goal={"target": "exploratory"},
        validation_policy={"mode": "walk_forward"},
    )
    assert build_input_fingerprints(**kwargs) == build_input_fingerprints(**kwargs)


# ---------------------------------------------------------------------------
# 37. Modification de la stratégie → seul strategy_sha256 change
# ---------------------------------------------------------------------------

def test_build_strategy_change_only_affects_strategy():
    df = _simple_df()
    goal = {"target": "exploratory"}
    policy = {"mode": "walk_forward"}

    base = build_input_fingerprints(dataframe=df, strategy={"fast": 10}, goal=goal, validation_policy=policy)
    modified = build_input_fingerprints(dataframe=df, strategy={"fast": 20}, goal=goal, validation_policy=policy)

    assert base["strategy_sha256"] != modified["strategy_sha256"]
    assert base["goal_sha256"] == modified["goal_sha256"]
    assert base["dataframe_sha256"] == modified["dataframe_sha256"]
    assert base["validation_policy_sha256"] == modified["validation_policy_sha256"]


# ---------------------------------------------------------------------------
# 38. Modification du goal → seul goal_sha256 change
# ---------------------------------------------------------------------------

def test_build_goal_change_only_affects_goal():
    df = _simple_df()
    strategy = {"fast": 10}
    policy = {"mode": "walk_forward"}

    base = build_input_fingerprints(dataframe=df, strategy=strategy, goal={"target": "exploratory"}, validation_policy=policy)
    modified = build_input_fingerprints(dataframe=df, strategy=strategy, goal={"target": "confirmatory"}, validation_policy=policy)

    assert base["goal_sha256"] != modified["goal_sha256"]
    assert base["strategy_sha256"] == modified["strategy_sha256"]
    assert base["dataframe_sha256"] == modified["dataframe_sha256"]
    assert base["validation_policy_sha256"] == modified["validation_policy_sha256"]


# ---------------------------------------------------------------------------
# 39. Modification du DataFrame → seul dataframe_sha256 change
# ---------------------------------------------------------------------------

def test_build_dataframe_change_only_affects_dataframe():
    strategy = {"fast": 10}
    goal = {"target": "exploratory"}
    policy = {"mode": "walk_forward"}

    df1 = _simple_df()
    df2 = _simple_df()
    df2.loc[0, "open"] = 99.0

    base = build_input_fingerprints(dataframe=df1, strategy=strategy, goal=goal, validation_policy=policy)
    modified = build_input_fingerprints(dataframe=df2, strategy=strategy, goal=goal, validation_policy=policy)

    assert base["dataframe_sha256"] != modified["dataframe_sha256"]
    assert base["strategy_sha256"] == modified["strategy_sha256"]
    assert base["goal_sha256"] == modified["goal_sha256"]
    assert base["validation_policy_sha256"] == modified["validation_policy_sha256"]


# ---------------------------------------------------------------------------
# 40. Pas de validation_policy → validation_policy_sha256 is None
# ---------------------------------------------------------------------------

def test_build_no_policy_gives_none():
    result = build_input_fingerprints(
        dataframe=_simple_df(),
        strategy={"name": "sma"},
        goal={"target": "exploratory"},
    )
    assert result["validation_policy_sha256"] is None


# ---------------------------------------------------------------------------
# 41. validation_policy présente → SHA-256 valide
# ---------------------------------------------------------------------------

def test_build_with_policy_gives_sha256():
    result = build_input_fingerprints(
        dataframe=_simple_df(),
        strategy={"name": "sma"},
        goal={"target": "exploratory"},
        validation_policy={"mode": "walk_forward", "windows": 5},
    )
    assert _is_sha256(result["validation_policy_sha256"])


# ---------------------------------------------------------------------------
# 42. Aucune clé interdite dans le résultat
# ---------------------------------------------------------------------------

FORBIDDEN_KEYS = {
    "best_strategy",
    "best_asset",
    "winner",
    "rank",
    "ranking",
    "global_score",
    "selected_strategy",
    "promotion",
    "auto_selection",
    "score",
    "profit_factor",
    "expectancy",
}


def test_build_no_forbidden_keys():
    result = build_input_fingerprints(
        dataframe=_simple_df(),
        strategy={"name": "sma"},
        goal={"target": "exploratory"},
        validation_policy={"mode": "walk_forward"},
    )
    found = set(result.keys()) & FORBIDDEN_KEYS
    assert not found, f"clés interdites trouvées : {found}"
