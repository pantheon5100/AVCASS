import math
import tempfile
import unittest
from pathlib import Path

from evaluate_wpr import infer_source, mean_or_nan, parse_method, read_class_map


class EvaluationHelpersTest(unittest.TestCase):
    def test_parse_method(self):
        name, path = parse_method("baseline=/tmp/example")
        self.assertEqual(name, "baseline")
        self.assertEqual(path, Path("/tmp/example"))

    def test_infer_source_from_filename_or_parent(self):
        self.assertEqual(infer_source(Path("clip/speech.wav")), "speech")
        self.assertEqual(infer_source(Path("clip/sfx/example.wav")), "sound effect")
        self.assertEqual(infer_source(Path("clip/unknown.wav")), None)

    def test_read_class_map(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "classes.csv"
            csv_path.write_text(
                "index,mid,display_name,main_class\n"
                "0,/a,Speech,speech\n"
                "1,/b,Music,music\n"
                "2,/c,Effect,sound effect\n"
                "3,/d,Silence,Silence\n",
                encoding="utf-8",
            )
            groups = read_class_map(csv_path)
        self.assertEqual(groups["speech"], [0])
        self.assertEqual(groups["Silence"], [3])

    def test_mean_or_nan(self):
        self.assertEqual(mean_or_nan([0.25, 0.75]), 0.5)
        self.assertTrue(math.isnan(mean_or_nan([])))


if __name__ == "__main__":
    unittest.main()
