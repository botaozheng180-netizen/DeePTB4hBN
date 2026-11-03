"""Run with python -m unittest discover -s tests -v (NumPy and ASE required)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conversion"))
from postscf_eigenvalues import parse_postscf_eigenvalues


def spectrum(k, offset=0):
    return (f"# k-point in recip. lattice units: {k} 0 0\n"
            "# state occupation number eigenvalue [eV]\n" +
            "\n".join(f"{i} {occ} {energy + offset}" for i, occ, energy in
                      [(1, 2, -200), (2, 2, -180), (3, 2, -10),
                       (4, 2, -8), (5, 2, -6), (6, 2, -4),
                       (7, 0, 2), (8, 0, 4)]) + "\n")


class DenseConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.case = self.root / "source" / "defect"
        self.case.mkdir(parents=True)
        (self.case / "geometry.in").write_text(
            "lattice_vector 3 0 0\nlattice_vector 0 3 0\nlattice_vector 0 0 20\n"
            "atom 0 0 0 B\natom 1 1 0 N\n")
        (self.case / "control.in").write_text("spin none\noutput postscf_eigenvalues\n")
        self.ks = self.case / "Final_KS_eigenvalues.dat"
        self.ks.write_text("# Number of k-points: 2\n# Number of states: 8\n" + spectrum(.5, 1) + spectrum(0))

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, script, *args):
        return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                              capture_output=True, text=True)

    def test_preserves_k_energy_pairs(self):
        k, occ, eig = parse_postscf_eigenvalues(self.ks)
        np.testing.assert_array_equal(k[:, 0], [.5, 0])
        np.testing.assert_array_equal(eig[:, 2], [-9, -10])
        np.testing.assert_array_equal(occ[:, -1], [0, 0])

    def test_rejects_corruption(self):
        original = self.ks.read_text()
        for bad in [original.rsplit("8 0 4", 1)[0],
                    original.replace("2 2 -179", "1 2 -179"),
                    original.replace("0.5 0 0", "1.0 0 0"),
                    original.replace("8 0 4", "8 0 nan"),
                    original.replace("8 0 4", "8 0 4 0 4")]:
            with self.subTest(text=bad[-60:]):
                self.ks.write_text(bad)
                with self.assertRaises(ValueError):
                    parse_postscf_eigenvalues(self.ks)

    def test_batch_resume_and_source_changes(self):
        out = self.root / "converted"
        args = [self.root / "source", out, "--eigenvalue-source", "postscf",
                "--band-policy", "all-noncore", "--charge", "0"]
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        dataset = out / "set.000000"
        eig = np.load(dataset / "eigenvalues.npy")
        self.assertEqual(eig.shape, (1, 2, 6))
        np.testing.assert_array_equal(eig[0, :, 0], [-9, -10])
        self.assertEqual(np.load(dataset / "occupations.npy").shape, eig.shape)
        report = json.loads((dataset / "conversion_report.json").read_text())
        self.assertEqual(report["kpoint_sampling"], "mesh")
        self.assertEqual(report["electron_count"]["modeled_electrons"], 8)
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args, "--resume")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.ks.write_text(self.ks.read_text() + "# changed\n")
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args, "--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing --resume", result.stdout + result.stderr)

    def test_fortran_exponents_and_header_counts(self):
        self.ks.write_text(self.ks.read_text().replace("0.5 0 0", "5.0D-1 0 0"))
        self.assertEqual(parse_postscf_eigenvalues(self.ks)[0][0, 0], .5)
        self.ks.write_text(self.ks.read_text().replace("Number of k-points: 2", "Number of k-points: 3"))
        with self.assertRaisesRegex(ValueError, "k-point count"):
            parse_postscf_eigenvalues(self.ks)

    def test_source_type_cannot_resume_path_as_mesh(self):
        self.test_legacy_band_route()
        out = self.root / "out"
        # A batch set with an existing path report must never become a mesh
        # solely because the new source file is present in the source folder.
        out.rename(self.root / "sets")
        sets = self.root / "sets"
        target = sets / "set.000000"
        target.mkdir()
        for path in list(sets.iterdir()):
            if path.is_file():
                path.rename(target / path.name)
        result = self.run_script("conversion/batch_aims_to_deeptb.py",
                                 self.root / "source", sets,
                                 "--eigenvalue-source", "postscf", "--resume",
                                 "--band-policy", "all-noncore")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing --resume", result.stdout + result.stderr)

    def test_spin_is_rejected(self):
        (self.case / "control.in").write_text("spin collinear\n")
        result = self.run_script("conversion/aims_to_deeptb.py", self.case,
                                 "-o", self.root / "out", "--eigenvalue-source", "postscf")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "out" / "eigenvalues.npy").exists())

    def test_optional_electron_count_and_occupations(self):
        for charge, expected in [(None, None), (0, 8), (1, 7)]:
            with self.subTest(charge=charge):
                out = self.root / f"charge_{charge}"
                args = [] if charge is None else ["--charge", charge]
                result = self.run_script("conversion/aims_to_deeptb.py", self.case,
                                         "-o", out, "--eigenvalue-source", "postscf",
                                         "--band-policy", "all-noncore", *args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((out / "conversion_report.json").read_text())
                self.assertEqual(report["electron_count"]["modeled_electrons"], expected)
                np.testing.assert_array_equal(np.load(out / "occupations.npy")[0],
                                              [[2, 2, 2, 2, 0, 0]] * 2)
                self.assertEqual(report["kpoint_coordinates"], "fractional_reciprocal")

    def test_resume_rejects_changed_control_and_missing_output(self):
        out = self.root / "converted"
        args = [self.root / "source", out, "--eigenvalue-source", "postscf",
                "--band-policy", "all-noncore"]
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        control = self.case / "control.in"
        original = control.read_text()
        control.write_text(original + "# changed settings\n")
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args, "--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing --resume", result.stdout + result.stderr)
        control.write_text(original)
        (out / "set.000000" / "occupations.npy").unlink()
        result = self.run_script("conversion/batch_aims_to_deeptb.py", *args, "--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing --resume", result.stdout + result.stderr)

    def test_mesh_qa_scatter_and_full_mode_guard(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / "visualization"))
        from visualize_band_regions import plot_one_set

        out = self.root / "mesh"
        result = self.run_script("conversion/aims_to_deeptb.py", self.case,
                                 "-o", out, "--eigenvalue-source", "postscf",
                                 "--band-policy", "all-noncore")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        options = dict(mode="selected", window_source="active", shade_color="orange",
                       k_tol=2e-5, aims_dir_override=None, dpi=72, linewidth=.7,
                       band_alpha=.5)
        image = self.root / "mesh_qa.png"
        with patch.object(plt, "close"):
            summary = plot_one_set(out, image, **options)
            ax = plt.gcf().axes[0]
            self.assertEqual(ax.get_xlabel(), "k-point index (mesh file order)")
            self.assertEqual(len(ax.collections), 6)
            self.assertFalse(any(len(line.get_xdata()) == 2 and
                                 list(line.get_xdata()) == [0, 1] and
                                 line.get_linestyle() == "-" for line in ax.lines))
        plt.close("all")
        self.assertEqual(summary["kpoint_sampling"], "mesh")
        self.assertGreater(image.stat().st_size, 0)
        options["mode"] = "full"
        with self.assertRaisesRegex(ValueError, "Dense meshes support --mode selected"):
            plot_one_set(out, image, **options)

    def test_legacy_band_route(self):
        rows = []
        for index, k in enumerate([0, .5], 1):
            data = " ".join(f"{occ} {energy}" for occ, energy in
                            [(2, -200), (2, -180), (2, -10), (2, -8),
                             (2, -6), (2, -4), (0, 2), (0, 4)])
            rows.append(f"{index} {k} 0 0 {data}\n")
        (self.case / "band1001.out").write_text("".join(rows))
        result = self.run_script("conversion/aims_to_deeptb.py", self.case,
                                 "-o", self.root / "out", "--band-policy", "all-noncore")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.root / "out" / "conversion_report.json").read_text())
        self.assertEqual(report["kpoint_sampling"], "path")


if __name__ == "__main__":
    unittest.main()
