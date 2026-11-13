"""Download and verification without network: local archives, tampering, unsafe archives."""

import hashlib
import io
import tarfile

import pytest

from flyvis_circuit_bench import download as DL
from flyvis_circuit_bench.cli import main


def _archive(package, path):
    with tarfile.open(path, "w:gz") as tar:
        tar.add(package, arcname=package.name)
    return f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"


def test_release_url_uses_owner_and_rejects_bad_names():
    url = DL.release_url("someone", DL.ASSET)
    assert url == f"https://github.com/someone/flyvis-circuit-bench/releases/download/v1.1.0/{DL.ASSET}"
    for bad in ("", "a/b", " x"):
        with pytest.raises(ValueError):
            DL.release_url(bad, DL.ASSET)


def test_fetch_refuses_plain_http(tmp_path):
    with pytest.raises(ValueError, match="https"):
        DL._fetch("http://example.org/x", tmp_path / "x")


def test_install_verifies_and_extracts(package, tmp_path):
    arch = tmp_path / DL.ASSET
    sums = _archive(package, arch)
    target = DL.install(arch, sums, tmp_path / "out")
    assert (target / "ablation" / "C1_000.npz").is_file()
    with pytest.raises(FileExistsError):
        DL.install(arch, sums, tmp_path / "out")


def test_install_rejects_checksum_mismatch(package, tmp_path):
    arch = tmp_path / DL.ASSET
    _archive(package, arch)
    with pytest.raises(ValueError, match="checksum mismatch"):
        DL.install(arch, f"{'0' * 64}  {arch.name}\n", tmp_path / "out")
    with pytest.raises(ValueError, match="not listed"):
        DL.install(arch, f"{'0' * 64}  other.tar.gz\n", tmp_path / "out")


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "pkg/../../evil.txt"])
def test_safe_extract_rejects_path_traversal(tmp_path, name):
    arch = tmp_path / "bad.tar.gz"
    with tarfile.open(arch, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))
    with pytest.raises(ValueError):
        DL.safe_extract(arch, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_rejects_links(tmp_path):
    arch = tmp_path / "link.tar.gz"
    with tarfile.open(arch, "w:gz") as tar:
        info = tarfile.TarInfo("pkg/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="not a regular file"):
        DL.safe_extract(arch, tmp_path / "out")


def test_cli_download_owner_defaults_to_the_release_account(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(DL, "download",
                        lambda owner, dest, repo, tag, version: calls.append((owner, version)) or tmp_path)
    monkeypatch.delenv("FCB_OWNER", raising=False)
    assert main(["download", "--dest", str(tmp_path)]) == 0
    monkeypatch.setenv("FCB_OWNER", "someone-else")
    assert main(["download", "--dest", str(tmp_path)]) == 0
    assert main(["download", "--owner", "third", "--dest", str(tmp_path)]) == 0
    assert calls == [("natiixnt", "1.1.0"), ("someone-else", "1.1.0"), ("third", "1.1.0")]
    assert main(["download", "--version", "1.0.0", "--dest", str(tmp_path)]) == 0
    assert calls[-1] == ("someone-else", "1.0.0")


def test_default_package_is_1_1_0_and_1_0_0_stays_available():
    assert DL.DATA_VERSION == "1.1.0" and DL.asset_name() == "measurements-v1.1.0.tar.gz"
    assert DL.asset_name("1.0.0") == "measurements-v1.0.0.tar.gz"
    with pytest.raises(ValueError, match="unknown data version"):
        DL.asset_name("2.0.0")


def test_download_builds_urls_for_the_requested_version(monkeypatch, tmp_path):
    urls = []
    monkeypatch.setattr(DL, "_fetch", lambda url, dest, timeout=60.0: urls.append(url) or dest.write_text(""))
    monkeypatch.setattr(DL, "install", lambda *a, **k: tmp_path)
    DL.download("someone", tmp_path, version="1.0.0")
    DL.download("someone", tmp_path)
    base = "https://github.com/someone/flyvis-circuit-bench/releases/download"
    assert urls == [f"{base}/v1.0.0/SHA256SUMS", f"{base}/v1.0.0/measurements-v1.0.0.tar.gz",
                    f"{base}/v1.1.0/SHA256SUMS", f"{base}/v1.1.0/measurements-v1.1.0.tar.gz"]


def test_cli_download_from_local_archive(package, tmp_path, capsys):
    arch = tmp_path / DL.ASSET
    (tmp_path / "SUMS").write_text(_archive(package, arch), encoding="utf-8")
    assert main(["download", "--archive", str(arch), "--sums", str(tmp_path / "SUMS"),
                 "--dest", str(tmp_path / "d")]) == 0
    assert "verified package" in capsys.readouterr().out
