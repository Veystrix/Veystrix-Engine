from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_REPOSITORY = "https://github.com/Veystrix/veystrix-engine"


class DistributionTests(unittest.TestCase):
    def test_readme_links_the_public_repository(self):
        readme = (ROOT / "README.md").read_text()
        self.assertIn(PUBLIC_REPOSITORY, readme)
        self.assertNotIn("Veystrix-Core", readme)

    def test_package_metadata_points_at_the_public_repository_only(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        urls = data["project"]["urls"]
        self.assertEqual(urls["Homepage"], PUBLIC_REPOSITORY)
        self.assertEqual(urls["Issues"], PUBLIC_REPOSITORY + "/issues")
        self.assertFalse([u for u in urls.values() if "Veystrix-Core" in u])
        changelog = (ROOT / "CHANGELOG.md").read_text()
        release = data["project"]["version"].split(".dev")[0]
        self.assertIn(f"## Veystrix {release}", changelog)

    def test_shared_runtime_dependencies_are_pinned_in_project_metadata(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            set(data["project"]["dependencies"]),
            {
                "mlx==0.32.2",
                "mlx-lm==0.31.3",
                "mlx-vlm==0.6.17",
                "transformers==5.16.1",
                "numpy==2.5.2",
                "requests==2.34.2",
                "huggingface-hub==1.29.0",
                "tomlkit==0.15.2",
            },
        )

    def test_ci_and_release_automation_are_separate(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
        self.assertIn("linux_unit_tests.py", ci)
        self.assertIn("secret_scan.py", ci)
        self.assertNotIn("gh release create", ci)
        self.assertIn("tags: ['v*']", release)
        for removed in ("apps/macos/", "packaging/macos/", "make_appcast.sh", "SUFeedURL", "swift test"):
            self.assertNotIn(removed, ci)
            self.assertNotIn(removed, release)
        self.assertNotIn("gh release create", release)
        self.assertNotIn("PUBLIC_RELEASES_TOKEN", release)
        self.assertNotIn("github.token", release)
        self.assertIn("permissions:\n  contents: read", release)


if __name__ == "__main__":
    unittest.main()
