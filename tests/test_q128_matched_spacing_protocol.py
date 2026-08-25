import contextlib
import io
import unittest

import torch

from training import reproducibility
from training.loss import (
    ECMLoss,
    Q128_MATCHED_SPACING_ARMS,
    Q128_MATCHED_SPACING_GAP_SCALE,
    Q128_MATCHED_SPACING_PROTOCOL,
    compute_target_weight_times,
    resolve_target_weight_factorial,
)
from training.schedules import get_schedule


Q128_FACTORS = {
    arm: factors for factors, arm in Q128_MATCHED_SPACING_ARMS.items()
}


class _RecordingDropoutNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.625))
        self.calls = []

    def forward(self, x, t, labels=None, augment_labels=None):
        del labels, augment_labels
        mask = (torch.rand_like(x) > 0.25).to(x.dtype)
        output = (x * self.weight + t) * mask
        self.calls.append(
            {
                "x": x.detach().clone(),
                "t": t.detach().clone(),
                "output": output.detach().clone(),
                "mask": mask.detach().clone(),
            }
        )
        return output


def _make_factorized(arm):
    target, denominator = Q128_FACTORS[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        loss = ECMLoss(
            q=128,
            k=8,
            b=1,
            c=0,
            adj="sigmoid",
            factorial_protocol=Q128_MATCHED_SPACING_PROTOCOL,
            target_gap_scale=target,
            denominator_gap_scale=denominator,
        )
    loss.update_schedule(0)
    return loss


def _make_native(arm):
    kwargs = dict(q=128, k=8, b=1, c=0)
    if arm == "A":
        kwargs.update(adj="sigmoid")
    elif arm == "Bsame":
        kwargs.update(adj="global_sigmoid", global_gap_scale=1.1)
    else:  # pragma: no cover - helper is intentionally A/Bsame-only.
        raise AssertionError(arm)
    with contextlib.redirect_stdout(io.StringIO()):
        loss = ECMLoss(**kwargs)
    loss.update_schedule(0)
    return loss


def _run_loss(loss, seed=20260824):
    net = _RecordingDropoutNet()
    images = torch.linspace(-1, 1, 4 * 2 * 3 * 3).reshape(4, 2, 3, 3)
    torch.manual_seed(seed)
    per_sample = loss(net=net, images=images)
    reduced = per_sample.mean()
    reduced.backward()
    return {
        "loss": per_sample.detach().clone(),
        "gradient": net.weight.grad.detach().clone(),
        "calls": net.calls,
        "telemetry": loss.factorial_runtime_metrics(),
        "post_rng": torch.get_rng_state().clone(),
    }


class Q128MatchedSpacingProtocolTest(unittest.TestCase):
    def test_exactly_five_frozen_arms_are_derived_from_factors(self):
        self.assertEqual(
            set(Q128_MATCHED_SPACING_ARMS.values()),
            {"A", "Bsame", "Bmatch", "Cmatch", "Dmatch"},
        )
        for factors, expected_arm in Q128_MATCHED_SPACING_ARMS.items():
            with self.subTest(arm=expected_arm):
                resolved = resolve_target_weight_factorial(
                    Q128_MATCHED_SPACING_PROTOCOL,
                    factors[0],
                    factors[1],
                    adj="sigmoid",
                    global_gap_scale=1.0,
                    q=128,
                    c=0,
                )
                self.assertEqual(resolved["arm"], expected_arm)

    def test_protocol_fails_closed_on_wrong_q_or_unfrozen_factor(self):
        invalid = [
            (256, 1.0, 1.0),
            (128, 0.56, 0.56),
            (128, 1.1, 1.0),
        ]
        for q, target, denominator in invalid:
            with self.subTest(q=q, target=target, denominator=denominator):
                with self.assertRaises(ValueError):
                    resolve_target_weight_factorial(
                        Q128_MATCHED_SPACING_PROTOCOL,
                        target,
                        denominator,
                        adj="sigmoid",
                        global_gap_scale=1.0,
                        q=q,
                        c=0,
                    )

    def test_matched_gap_tracks_q256_reference_at_stage_zero(self):
        t = torch.logspace(-5, 3, 10000, dtype=torch.float64).reshape(-1, 1, 1, 1)
        q256_r = get_schedule("sigmoid", q=256, k=8, b=1).compute_r(t, stage=0)
        q128_r = get_schedule("sigmoid", q=128, k=8, b=1).compute_r(t, stage=0)
        _, _, q256_gap, _ = compute_target_weight_times(
            t,
            q256_r,
            target_gap_scale=1.1,
            denominator_gap_scale=1.1,
        )
        _, _, q128_gap, _ = compute_target_weight_times(
            t,
            q128_r,
            target_gap_scale=Q128_MATCHED_SPACING_GAP_SCALE,
            denominator_gap_scale=Q128_MATCHED_SPACING_GAP_SCALE,
        )
        self.assertTrue(torch.allclose(q128_gap, q256_gap, rtol=1e-11, atol=1e-15))

    def test_stage_zero_matched_state_and_inverse_gap_identities(self):
        t = torch.logspace(-5, 3, 10000, dtype=torch.float64).reshape(-1, 1, 1, 1)
        base_r = get_schedule("sigmoid", q=128, k=8, b=1).compute_r(t, stage=0)
        states = {}
        for arm, (target, denominator) in Q128_FACTORS.items():
            states[arm] = compute_target_weight_times(
                t,
                base_r,
                target_gap_scale=target,
                denominator_gap_scale=denominator,
            )

        self.assertTrue(torch.equal(states["A"][0], states["Dmatch"][0]))
        self.assertTrue(torch.equal(states["Bmatch"][0], states["Cmatch"][0]))
        self.assertEqual(
            reproducibility.state_sha256(states["A"][0]),
            reproducibility.state_sha256(states["Dmatch"][0]),
        )
        self.assertEqual(
            reproducibility.state_sha256(states["Bmatch"][0]),
            reproducibility.state_sha256(states["Cmatch"][0]),
        )
        g_a = states["A"][3].reciprocal()
        g_d = states["Dmatch"][3].reciprocal()
        g_bmatch = states["Bmatch"][3].reciprocal()
        g_cmatch = states["Cmatch"][3].reciprocal()
        # The identity is analytic; the production path multiplies the gap and
        # subtracts from t before the reciprocal, so float64 roundoff is not
        # bitwise.  The observed worst relative error is below 3e-14.
        torch.testing.assert_close(g_d, g_a / 0.55, rtol=5e-14, atol=1e-12)
        torch.testing.assert_close(
            g_bmatch, g_cmatch / 0.55, rtol=5e-14, atol=1e-12
        )
        self.assertTrue(all(bool((state[0] > 0).all()) for state in states.values()))
        self.assertTrue(all(bool((state[1] > 0).all()) for state in states.values()))

    def test_full_loss_target_hash_and_gradient_scaling_identities(self):
        runs = {arm: _run_loss(_make_factorized(arm)) for arm in Q128_FACTORS}
        for left, right in (("A", "Dmatch"), ("Bmatch", "Cmatch")):
            self.assertEqual(
                runs[left]["telemetry"]["target_r_sha256"],
                runs[right]["telemetry"]["target_r_sha256"],
            )
            self.assertEqual(
                runs[left]["telemetry"]["target_delta_sha256"],
                runs[right]["telemetry"]["target_delta_sha256"],
            )
            for left_call, right_call in zip(runs[left]["calls"], runs[right]["calls"]):
                self.assertTrue(torch.equal(left_call["x"], right_call["x"]))
                self.assertTrue(torch.equal(left_call["t"], right_call["t"]))
                self.assertTrue(torch.equal(left_call["mask"], right_call["mask"]))
                self.assertTrue(torch.equal(left_call["output"], right_call["output"]))
            self.assertTrue(torch.equal(runs[left]["post_rng"], runs[right]["post_rng"]))
        torch.testing.assert_close(
            runs["Dmatch"]["loss"], runs["A"]["loss"] / 0.55,
            rtol=2e-6, atol=1e-6,
        )
        torch.testing.assert_close(
            runs["Dmatch"]["gradient"], runs["A"]["gradient"] / 0.55,
            rtol=2e-6, atol=1e-6,
        )
        torch.testing.assert_close(
            runs["Bmatch"]["loss"], runs["Cmatch"]["loss"] / 0.55,
            rtol=2e-6, atol=1e-6,
        )
        torch.testing.assert_close(
            runs["Bmatch"]["gradient"], runs["Cmatch"]["gradient"] / 0.55,
            rtol=2e-6, atol=1e-6,
        )

    def test_a_and_bsame_match_native_paths(self):
        for arm in ("A", "Bsame"):
            with self.subTest(arm=arm):
                factorized = _run_loss(_make_factorized(arm))
                native = _run_loss(_make_native(arm))
                self.assertTrue(torch.equal(factorized["loss"], native["loss"]))
                self.assertTrue(torch.equal(factorized["gradient"], native["gradient"]))
                self.assertTrue(torch.equal(factorized["post_rng"], native["post_rng"]))
                for factorized_call, native_call in zip(
                    factorized["calls"], native["calls"]
                ):
                    self.assertTrue(torch.equal(factorized_call["x"], native_call["x"]))
                    self.assertTrue(torch.equal(factorized_call["t"], native_call["t"]))
                    self.assertTrue(torch.equal(factorized_call["mask"], native_call["mask"]))
                    self.assertTrue(
                        torch.equal(factorized_call["output"], native_call["output"])
                    )


if __name__ == "__main__":
    unittest.main()
