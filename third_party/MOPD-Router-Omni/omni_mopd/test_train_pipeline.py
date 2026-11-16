"""Safety checks for the job wrapper, not changes to native optimization."""
import math
import unittest

from omni_mopd.train_pipeline import parse_metrics


class PipelineTests(unittest.TestCase):
    def test_native_ray_console_line(self):
        row = parse_metrics(
            "\x1b[36m(TaskRunner pid=5)\x1b[0m step:2 - actor/grad_norm:1.5e-3"
            " - actor/delta_router/selected_token_ratio:0.8"
            " - training/global_step:2 - response_length/mean:100"
        )
        self.assertEqual(row["training/global_step"], 2)
        self.assertEqual(row["actor/grad_norm"], .0015)
        self.assertEqual(row["response_length/mean"], 100)

    def test_nonfinite_not_silently_ignored(self):
        row = parse_metrics("step:1 - actor/grad_norm:nan - actor/pg_loss:inf")
        self.assertFalse(math.isfinite(row["actor/grad_norm"]))
        self.assertFalse(math.isfinite(row["actor/pg_loss"]))

    def test_unrelated_logs_ignored(self):
        self.assertIsNone(parse_metrics("Loading weights: 100%"))
        self.assertIsNone(parse_metrics("critic/rewards/mean:0"))


if __name__ == "__main__":
    unittest.main()
