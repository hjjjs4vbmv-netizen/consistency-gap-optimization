import unittest

import torch

from scripts.classify_m1_readout import classify_fixed_input


class ToyReadout(torch.nn.Module):
    img_resolution = 4
    img_channels = 3
    label_dim = 0

    def __init__(self, *, bad_state=False, bad_output=False, raise_on_forward=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        if bad_state:
            self.weight.data.fill_(float("nan"))
        self.bad_output = bad_output
        self.raise_on_forward = raise_on_forward

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        self.last_call = {
            "x": x.detach().clone(), "sigma": sigma.detach().clone(),
            "class_labels": class_labels, "force_fp32": force_fp32,
            "training": self.training, "grad_enabled": torch.is_grad_enabled(),
        }
        if self.raise_on_forward:
            raise RuntimeError("deliberate forward failure")
        if self.bad_output:
            return torch.full_like(x, float("inf"))
        return x + self.weight


class TestM1ReadoutClassifier(unittest.TestCase):
    def test_finite_readout_runs_frozen_fixed_input_forward(self):
        model = ToyReadout()
        result = classify_fixed_input(model, torch.device("cpu"))
        self.assertEqual(result["classification"], "FINITE_READOUT")
        self.assertEqual(result["output_nonfinite_count"], 0)
        self.assertEqual(result["invalid_fields"], [])
        self.assertTrue(torch.equal(model.last_call["x"], torch.zeros(1, 3, 4, 4)))
        self.assertTrue(torch.equal(model.last_call["sigma"], torch.ones(1)))
        self.assertIsNone(model.last_call["class_labels"])
        self.assertTrue(model.last_call["force_fp32"])
        self.assertFalse(model.last_call["training"])
        self.assertFalse(model.last_call["grad_enabled"])

    def test_nonfinite_state_and_output_are_observed_not_inferred(self):
        state = classify_fixed_input(
            ToyReadout(bad_state=True), torch.device("cpu")
        )
        self.assertEqual(
            state["classification"],
            "NONFINITE_READOUT_STATE_AND_FIXED_OUTPUT",
        )
        self.assertEqual(state["nonfinite_state_tensor_paths"], ["weight"])
        self.assertGreater(state["output_nonfinite_count"], 0)

        output = classify_fixed_input(
            ToyReadout(bad_output=True), torch.device("cpu")
        )
        self.assertEqual(
            output["classification"], "NONFINITE_FIXED_INPUT_OUTPUT"
        )
        self.assertEqual(output["nonfinite_state_tensor_paths"], [])
        self.assertEqual(output["output_nonfinite_count"], 48)

        direct = classify_fixed_input(
            ToyReadout(bad_state=True, raise_on_forward=True),
            torch.device("cpu"),
        )
        self.assertEqual(direct["classification"], "NONFINITE_READOUT_STATE")
        self.assertFalse(direct["fixed_input_executed"])
        self.assertIsNone(direct["output_nonfinite_count"])
        self.assertEqual(
            direct["fixed_input_forward_error"]["type"], "RuntimeError"
        )


if __name__ == "__main__":
    unittest.main()
