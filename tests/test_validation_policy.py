"""Tests de structure de config/validation_policy.yaml.

Charge le YAML directement avec yaml.safe_load.
Aucun loader src/ requis. Aucun accès réseau.
"""

from pathlib import Path

import pytest
import yaml

POLICY_PATH = Path(__file__).parent.parent / "config" / "validation_policy.yaml"


@pytest.fixture(scope="module")
def policy():
    with open(POLICY_PATH) as f:
        return yaml.safe_load(f)


def test_fichier_existe_et_est_un_dict(policy):
    assert isinstance(policy, dict)


def test_schema_version_present(policy):
    assert "schema_version" in policy["protocol"]


def test_policy_version_present(policy):
    assert "policy_version" in policy["protocol"]


def test_statut_initial_draft(policy):
    assert policy["protocol"]["status"] == "draft"


def test_frozen_at_est_none(policy):
    assert policy["protocol"]["frozen_at"] is None


def test_protocol_hash_est_none(policy):
    assert policy["protocol"]["protocol_hash"] is None


def test_numeric_thresholds_frozen_est_false(policy):
    assert policy["protocol"]["numeric_thresholds_frozen"] is False


def test_quatre_statuts_de_decision(policy):
    expected = {
        "not_evaluable",
        "exploratory_only",
        "fails_current_protocol",
        "eligible_for_extended_paper_validation",
    }
    assert set(policy["decision_statuses"]) == expected


def test_quatre_roles_de_donnees(policy):
    expected = {
        "exploratory_data",
        "confirmatory_holdout",
        "prospective_holdout_candidate",
        "paper_forward",
    }
    assert set(policy["data_roles"]) == expected


def test_statuts_futurs_holdout_presents(policy):
    expected = {
        "reserved",
        "accumulating",
        "opened",
        "consumed",
        "contaminated",
        "retired",
    }
    actual = set(policy["holdout_budget_policy"]["allowed_future_statuses"])
    assert actual == expected


def test_datasets_est_liste_vide(policy):
    assert policy["holdout_budget_policy"]["datasets"] == []


def test_ouverture_holdout_irreversible(policy):
    assert policy["holdout_budget_policy"]["opening_is_irreversible"] is True


def test_paper_forward_desactive(policy):
    assert policy["paper_forward"]["enabled"] is False


def test_early_pass_allowed_est_false(policy):
    assert policy["paper_forward"]["early_pass_allowed"] is False


def test_strategy_changes_during_run_est_false(policy):
    assert policy["paper_forward"]["strategy_changes_during_run"] is False


def test_seuils_numeriques_paper_forward_sont_none(policy):
    pf = policy["paper_forward"]
    assert pf["minimum_calendar_duration"] is None
    assert pf["minimum_observations"] is None
    assert pf["minimum_trades"] is None


def test_champs_puissance_statistique_sont_none(policy):
    sp = policy["statistical_power"]
    assert sp["target_power"] is None
    assert sp["significance_level"] is None
    assert sp["minimum_detectable_effect"] is None


def test_insufficient_power_reason(policy):
    assert (
        policy["statistical_power"]["insufficient_power_reason"]
        == "underpowered_for_declared_effect"
    )


def test_dsr_desactive_et_not_computed(policy):
    dsr = policy["statistical_diagnostics"]["dsr"]
    assert dsr["enabled"] is False
    assert dsr["applicability"] == "not_computed"


def test_pbo_desactive_et_not_computed(policy):
    pbo = policy["statistical_diagnostics"]["pbo"]
    assert pbo["enabled"] is False
    assert pbo["applicability"] == "not_computed"


def test_gate_conjunctif_desactive(policy):
    assert policy["future_conjunctive_gate"]["enabled"] is False


def test_numeric_thresholds_est_none(policy):
    assert policy["future_conjunctive_gate"]["numeric_thresholds"] is None


def test_aggregation_method(policy):
    assert (
        policy["future_conjunctive_gate"]["aggregation_method"]
        == "all_required_conditions"
    )
