import tomllib
from pathlib import Path

from black_hole import __version__


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)["project"]

    assert project["version"] == __version__

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog
    assert f"releases/tag/v{__version__}" in changelog
