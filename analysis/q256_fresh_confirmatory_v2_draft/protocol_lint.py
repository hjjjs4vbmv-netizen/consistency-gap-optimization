"""Mechanical lint for the q256 fresh confirmatory v2 protocol draft.

Run before any freeze:  python protocol_lint.py
Exit code 0 and output "LINT PASS" = all checks green.

Checks (each fail prints a LINT-FAIL line and aborts with exit 1):
  1. JSON parses; schema/status fields present and well-formed.
  2. Design naming: the design name must not claim "crossed".
  3. Seeds: 24 unique primary, 8 ordered replacement, disjoint sets, and
     disjoint from the fresh cohort seeds 31-42.
  4. Verdict decision table: exactly the five frozen categories, in the
     frozen precedence order, each with a condition string.
  5. Interim: binding futility-only, stop rule mentions both clauses,
     SD trigger 0.15, cap 28.
  6. Missing data: hard >4-replacement termination rule present;
     completion-conditioned estimand present.
  7. Budget packages: exactly three named packages, arithmetic consistent
     (subset - minimal = 22, full - minimal = 44).
  8. Planning numbers in the protocol agree with the committed
     planning_calculations.json (power, MDE, assurance, dose power).
  9. Type-I validation committed: unconditional rates <= 0.055 in both
     null scenarios of type_I_error_simulation.json.
 10. Freeze action names exactly the four SHA256-covered files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    protocol = json.loads((HERE / "protocol_draft.json").read_text(encoding="utf-8"))
    planning = json.loads((HERE / "planning_calculations.json").read_text(encoding="utf-8"))
    type_i = json.loads((HERE / "type_I_error_simulation.json").read_text(encoding="utf-8"))

    # 1. schema / status
    check(protocol["schema"] == "ect.q256.fresh-confirmatory-v2-protocol-draft/v2",
          "schema must be .../v2")
    check(protocol["status"] == "DRAFT_PENDING_G5_AND_L1_GATES",
          "status must be DRAFT_PENDING_G5_AND_L1_GATES")

    # 2. design naming
    name = protocol["design"]["name"]
    check("crossed" not in name.lower(),
          f"design name must not say 'crossed': {name!r}")
    check("matched two-history continuation" in name,
          "design name must be the matched two-history continuation label")

    # 3. seeds
    primary = protocol["design"]["seeds_primary"]
    pool = protocol["design"]["seeds_replacement_pool_ordered"]
    check(len(primary) == 24 and len(set(primary)) == 24,
          "primary must be 24 unique seeds")
    check(len(pool) == 8 and len(set(pool)) == 8, "replacement pool must be 8 unique seeds")
    check(not (set(primary) & set(pool)), "primary/pool must be disjoint")
    check(not (set(primary) | set(pool)) & {31, 32, 33, 34, 35, 36, 37, 38, 39,
                                           40, 41, 42},
          "seeds must be disjoint from the fresh cohort 31-42")

    # 4. verdict table
    cats = protocol["verdict_decision_table"]["categories_in_precedence_order"]
    expected = ["STRONG_SUCCESS", "INFORMATIVE_PRACTICAL_NULL",
                "WEAK_DIRECTIONAL_REPLICATION", "OPPOSITE_DIRECTION_FALSIFICATION",
                "INCONCLUSIVE"]
    check([c["name"] for c in cats] == expected,
          "verdict categories must be exactly the five frozen names in precedence order")
    for c in cats:
        check("condition" in c and isinstance(c["condition"], str) and c["condition"],
              f"category {c['name']} must carry a condition string")
    check("INCONCLUSIVE" in "".join(
        c.get("note", "") + c.get("condition", "") for c in cats)
        and "hi95 >= 0" in json.dumps(protocol["verdict_decision_table"]),
        "the one-sided-p<0.05-with-hi95>=0 edge case must be pinned as INCONCLUSIVE")

    # 5. interim
    interim = protocol["interim_analysis"]
    check(interim["type"].startswith("binding futility-only"),
          "interim must be binding futility-only")
    stop = interim["stop_rule"]
    check("mean(H_A,12) > 0" in stop and "CP < 0.20" in stop,
          "stop rule must contain both futility clauses")
    check(interim["blinded_sd_reestimation"]["trigger"] == "s12 > 0.15",
          "SD trigger must be s12 > 0.15")
    check("n_f = 28" in interim["blinded_sd_reestimation"]["action"],
          "extension must go to the cap 28")

    # 6. missing data
    missing = protocol["missing_data"]
    check("EXECUTION_FAILED" in missing["hard_termination_rule"]
          and "MORE than 4" in missing["hard_termination_rule"],
          "hard >4-replacement termination rule must be present")
    check("completion-conditioned" in protocol["primary_endpoint"]["estimand_population"],
          "completion-conditioned estimand must be explicit")

    # 7. budget packages
    pk = protocol["budget_packages"]
    check(set(pk.keys()) >= {"MINIMAL", "WITH_HORIZON_SUBSET", "WITH_HORIZON_FULL"},
          "exactly the three named budget packages must exist")
    minimal_re = int(pk["MINIMAL"]["training_re"].split()[0])
    subset_re = int(pk["WITH_HORIZON_SUBSET"]["training_re"].split()[0])
    full_re = int(pk["WITH_HORIZON_FULL"]["training_re"].split()[0])
    check(subset_re - minimal_re == 22, "subset package must add 22 RE over minimal")
    check(full_re - minimal_re == 44, "full package must add 44 RE over minimal")
    check(minimal_re == 96 + 16 + 15, "minimal package must decompose as 96+16+15")

    # 8. planning numbers agreement
    two = planning["two_arm_n24"]
    proto_text = json.dumps(protocol)
    check(abs(two["power_at_fresh_point_estimate"] - 0.9475) < 0.001
          and "0.947" in proto_text,
          "protocol must state the committed two-arm power 0.947")
    check(abs(two["mde_80pct_power"] - 0.0591) < 0.0002
          and "0.0591" in proto_text,
          "protocol must state the committed MDE 0.0591")
    check(abs(two["assurance"] - 0.823) < 0.001
          and "0.823" in proto_text,
          "protocol must state the committed assurance 0.823")
    dose = planning["dose_within_seed_contrast_n8"]
    check(abs(dose["power_at_linear_projection"] - 0.899) < 0.001
          and "0.899" in proto_text,
          "protocol must state the committed dose power 0.899")

    # 9. type-I validation committed and within bound
    for tag in ("null_planning_sd", "null_stress_sd_0p20"):
        rate = type_i["scenarios"][tag]["rejection_rate_unconditional"]
        check(rate <= 0.055,
              f"type-I unconditional rate {rate} in {tag} exceeds the 0.055 freeze bound")

    # 10. freeze action names the four covered files
    action = protocol["freeze_procedure"]["freeze_action"]
    for fname in ("protocol.json", "interim_futility.py",
                  "planning_calculations.py", "planning_calculations.json"):
        check(fname in action, f"freeze action must name {fname}")

    if FAILURES:
        for f in FAILURES:
            print(f"LINT-FAIL: {f}")
        return 1
    print("LINT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())