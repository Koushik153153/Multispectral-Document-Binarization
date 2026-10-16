"""Tiny byte fixtures: packaging tests require no ML libraries or real dataset."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_gpu_training.py"
SPEC = importlib.util.spec_from_file_location("package_gpu_training", SCRIPT)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def put(root, name, data=b"fixture\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for name in package.REQUIRED:
        put(root, name, b"# fixture\n")
    put(root, "splits/train_keys.txt", b"A\r\n")
    put(root, "splits/val_keys.txt", b" B \n\n")
    for key in ("A", "B"):
        for band in range(12):
            put(root, f"MSBin/train/images/{key}_{band}.png", f"{key}:{band}".encode())
        put(root, f"MSBin/train/labels/{key}.png", key.encode())
    return root


def output(root):
    return root / "transfer" / "training.zip"


def rewrite_archive(path, transform):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    transform(entries)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_selection_exclusions_and_byte_preservation(project):
    excluded = ["MSBin/test/images/T_0.png", "splits/test_keys.txt",
                "MSBin/train_val_split/train/images/A_0.png", ".venv/secret.py",
                "__pycache__/cache.pyc", "AMDES_IMPLEMENTATION/code.py", ".git/config",
                ".env", "secrets.json", "runs_cpu/pilot/best.pt",
                "MSBin/train/images/weights.pt", "predictions/BT16.png",
                "old.zip", "transfer/old.zip", "unrelated.py"]
    for name in excluded:
        put(project, name, b"excluded")
    for name in package.OPTIONAL:
        put(project, name, b"historical provenance\r\n")
    before = {p.relative_to(project).as_posix(): p.read_bytes()
              for p in project.rglob("*") if p.is_file()}
    result = package.create_archive(project, output(project))
    assert result["verified"]
    assert result["split_counts"] == {"train": 1, "val": 1, "original_train": 2, "test_packaged": 0}
    with zipfile.ZipFile(output(project)) as archive:
        names = set(archive.namelist())
        assert not names.intersection(excluded)
        assert names == {name for name in before if package.allowed(name)} | {package.MANIFEST}
        for name in names - {package.MANIFEST}:
            assert archive.read(name) == before[name]
        manifest = json.loads(archive.read(package.MANIFEST))
        assert manifest["exclusions"] == package.EXCLUSIONS
        assert "historical" in manifest["audit_provenance"]
    assert all((project / name).read_bytes() == data for name, data in before.items())


@pytest.mark.parametrize("missing", ["training_support.py", "splits/val_keys.txt",
                                     "MSBin/train/images/A_11.png", "MSBin/train/labels/B.png"])
def test_missing_files_fail_without_publication(project, missing):
    (project / missing).unlink()
    with pytest.raises(ValueError):
        package.create_archive(project, output(project))
    assert not output(project).exists()
    assert not list(project.rglob("*.tmp"))


def test_overwrite_requires_explicit_flag(project):
    path = output(project)
    package.create_archive(project, path)
    original = path.read_bytes()
    put(project, "README.md", b"changed source")
    with pytest.raises(FileExistsError):
        package.create_archive(project, path)
    assert path.read_bytes() == original
    package.create_archive(project, path, overwrite=True)
    assert package.verify_archive(path)["verified"]
    with zipfile.ZipFile(path) as archive:
        assert archive.read("README.md") == b"changed source"


def test_verification_needs_only_archive_and_does_not_extract(project, tmp_path):
    path = tmp_path / "delivery" / "training.zip"
    package.create_archive(project, path)
    (project / "splits/train_keys.txt").unlink()
    before = set(tmp_path.rglob("*"))
    assert package.verify_archive(path)["verified"]
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("damage", ["hash", "extra", "missing", "counts", "manifest"])
def test_verification_rejects_damage(project, damage):
    path = output(project)
    package.create_archive(project, path)

    def corrupt(entries):
        if damage == "hash":
            original = entries["README.md"]
            entries["README.md"] = b"!" + original[1:]  # Same size, wrong hash.
        elif damage == "extra":
            entries["MSBin/test/images/T_0.png"] = b"forbidden"
        elif damage == "missing":
            del entries["msbin_labels.py"]
        elif damage == "manifest":
            del entries[package.MANIFEST]
        else:
            manifest = json.loads(entries[package.MANIFEST])
            manifest["split_counts"]["train"] = 99
            entries[package.MANIFEST] = json.dumps(manifest).encode()

    rewrite_archive(path, corrupt)
    with pytest.raises(ValueError):
        package.verify_archive(path)


def test_collision_and_band_alias_rejected(project):
    with pytest.raises(ValueError, match="collision"):
        package.unique_names(["source.py", "SOURCE.py"])
    put(project, "MSBin/train/images/A_00.png")
    with pytest.raises(ValueError, match="Duplicate band alias"):
        package.create_archive(project, output(project))


def test_unlisted_local_import_rejected(project):
    put(project, "training_support.py", b"import extra_dependency\n")
    put(project, "extra_dependency.py", b"# local module\n")
    with pytest.raises(ValueError, match="dependency not allowlisted"):
        package.create_archive(project, output(project))


def test_failed_verification_preserves_previous_archive(project, monkeypatch):
    path = output(project)
    put(project, "transfer/training.zip", b"existing archive")

    def fail(_):
        raise ValueError("injected verification failure")

    monkeypatch.setattr(package, "verify_archive", fail)
    with pytest.raises(ValueError, match="injected"):
        package.create_archive(project, path, overwrite=True)
    assert path.read_bytes() == b"existing archive"
    assert not list(path.parent.glob("*.tmp"))


def test_no_clobber_if_destination_appears_during_build(project, monkeypatch):
    path = output(project)
    verify = package.verify_archive

    def competing_writer(temporary):
        result = verify(temporary)
        path.write_bytes(b"another writer")
        return result

    monkeypatch.setattr(package, "verify_archive", competing_writer)
    with pytest.raises(FileExistsError):
        package.create_archive(project, path)
    assert path.read_bytes() == b"another writer"
    assert not list(path.parent.glob("*.tmp"))


def test_streaming_uses_bounded_reads():
    class BoundedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= package.CHUNK
            return super().read(size)

    destination = io.BytesIO()
    size, digest = package.stream_hash(BoundedReader(b"small fixture"), destination)
    assert size == len(b"small fixture")
    assert len(digest) == 64
    assert destination.getvalue() == b"small fixture"


def test_output_cannot_be_inside_dataset(project):
    with pytest.raises(ValueError, match="Output directory cannot"):
        package.create_archive(project, project / "MSBin" / "transfer" / "training.zip")


def test_verified_extraction_requires_new_directory(project, tmp_path):
    path = output(project)
    package.create_archive(project, path)
    destination = tmp_path / "extracted"
    assert package.extract_archive(path, destination)["verified"]
    assert (destination / "splits/train_keys.txt").read_bytes() == b"A\r\n"
    with pytest.raises(FileExistsError):
        package.extract_archive(path, destination)


def test_unsafe_archive_rejected_before_extraction(project, tmp_path):
    path = output(project)
    package.create_archive(project, path)
    rewrite_archive(path, lambda entries: entries.update({"../escape.txt": b"outside"}))
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe"):
        package.extract_archive(path, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()
