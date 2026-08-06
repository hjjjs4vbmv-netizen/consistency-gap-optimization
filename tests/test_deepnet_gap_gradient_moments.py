import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "deepnet_gap_gradient_moments.py"
SPEC = importlib.util.spec_from_file_location("deepnet_gap_gradient_moments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DeepnetGapGradientMomentTests(unittest.TestCase):
    def test_parse_gaps_requires_reference(self):
        self.assertEqual(MODULE.parse_gaps("0.9,1.0,1.2"), [0.9, 1.0, 1.2])
        with self.assertRaises(Exception):
            MODULE.parse_gaps("0.9,1.2")

    def test_projection_moments(self):
        # Mean gradients: mu_1=[1,0], mu_2=[2,2].  Thus a*=2 and residual
        # sqrt(4)/sqrt(8)=1/sqrt(2); batch norm identity gives variance=0.
        means = {
            "1.0": {"layer.weight": MODULE.torch.tensor([2.0, 0.0])},
            "2.0": {"layer.weight": MODULE.torch.tensor([4.0, 4.0])},
        }
        rows, layers = MODULE.mean_vector_statistics(
            means, 1.0, 2,
            {"1.0": 1.0, "2.0": 8.0},
            {"1.0": {"layer": 1.0}, "2.0": {"layer": 8.0}},
        )
        by_gap = {row["gap"]: row for row in rows}
        self.assertAlmostEqual(by_gap[2.0]["scalar_fit_to_g1"], 2.0)
        self.assertAlmostEqual(by_gap[2.0]["direction_residual"], 2 ** -0.5)
        self.assertAlmostEqual(by_gap[2.0]["gradient_variance_trace"], 0.0)
        self.assertEqual(len(layers), 2)


if __name__ == "__main__":
    unittest.main()
