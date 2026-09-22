"""Apply the versioned PSF portability patch; never reset an edited submodule."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PSF = ROOT / "third_party" / "PSF"
PATCH = ROOT / "patches" / "psf_windows.patch"
PSF_COMMIT = "c74b39e1200513039cfb8d776505fb75da599e68"


def git(*args):
    return subprocess.run(["git", "-C", str(PSF), *args], capture_output=True, text=True)


def check_version():
    if not (PSF / "modules" / "functional" / "backend.py").is_file():
        raise RuntimeError("PSF missing. Run: git submodule update --init --recursive")
    result = git("rev-parse", "HEAD")
    if result.returncode or result.stdout.strip() != PSF_COMMIT:
        raise RuntimeError(f"Expected PSF {PSF_COMMIT}; got {result.stdout.strip()}\n{result.stderr}")


def provenance():
    """Require preparation before importing any JIT-compiled upstream code."""
    check_version()
    result = git("apply", "--reverse", "--check", str(PATCH))
    if result.returncode:
        raise RuntimeError("PSF patch missing or conflicting. Run: python prepare_psf.py")
    diff = git("diff", "--no-ext-diff", "HEAD", "--")
    if diff.returncode:
        raise RuntimeError(diff.stderr)
    return {"repository": "https://github.com/klightz/PSF", "commit": PSF_COMMIT,
            "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "working_diff_sha256": hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest()}


def main(check=False):
    check_version()
    if git("apply", "--reverse", "--check", str(PATCH)).returncode == 0:
        print("PSF patch already applied.")
    elif check:
        raise RuntimeError("PSF is not prepared. Run: python prepare_psf.py")
    else:
        result = git("apply", "--check", str(PATCH))
        if result.returncode:
            raise RuntimeError("Patch conflicts; existing edits were preserved.\n" + result.stderr)
        result = git("apply", str(PATCH))
        if result.returncode:
            raise RuntimeError(result.stderr)
        print("PSF patch applied.")
    print(provenance())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(**vars(parser.parse_args()))
