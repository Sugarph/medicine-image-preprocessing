from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def test_installed_setuptools_build_contains_and_loads_v2_threshold_resource(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    source_copy = tmp_path / "source"
    shutil.copytree(
        repository,
        source_copy,
        ignore=shutil.ignore_patterns(".git", ".venv", ".pytest_cache", ".superpowers", "__pycache__"),
    )
    install_dir = tmp_path / "installed"
    build = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from setuptools.dist import Distribution; "
                "from setuptools.config.pyprojecttoml import apply_configuration; "
                "import sys; "
                "source=Path(sys.argv[1]); out=Path(sys.argv[2]); "
                "dist=Distribution(); dist.script_name=str(source/'setup.py'); "
                "apply_configuration(dist, str(source/'pyproject.toml')); "
                "cmd=dist.get_command_obj('build_py'); cmd.build_lib=str(out); "
                "cmd.ensure_finalized(); cmd.run()"
            ),
            str(source_copy),
            str(install_dir),
        ],
        cwd=source_copy,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    assert (install_dir / "medicine_preprocess" / "data" / "__init__.py").exists()
    assert (install_dir / "medicine_preprocess" / "data" / "quality_thresholds_v2.json").exists()

    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(install_dir)!r}); "
                "from medicine_preprocess import PreprocessConfig; "
                "assert PreprocessConfig.grabcut().preset_version == '1'"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
