import importlib.util
import os
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ROOT_DIR, ".vendor312")
REQUIREMENTS_FILE = os.path.join(ROOT_DIR, "requirements.txt")

REQUIRED_MODULES = {
    "flask": "Flask",
    "cv2": "opencv-contrib-python",
    "pandas": "pandas",
    "numpy": "numpy",
    "PIL": "Pillow",
    "openpyxl": "openpyxl",
}

OPTIONAL_MODULES = {
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
}


def ensure_vendor_path():
    os.makedirs(VENDOR_DIR, exist_ok=True)
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)


def missing_modules():
    missing = []
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def missing_optional_modules():
    missing = []
    for module_name in OPTIONAL_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def install_missing_packages():
    env = os.environ.copy()
    env["PYTHONPATH"] = VENDOR_DIR + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--target",
        VENDOR_DIR,
        "-r",
        REQUIREMENTS_FILE,
    ]
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def main():
    ensure_vendor_path()
    initial_missing = missing_modules()
    initial_optional_missing = missing_optional_modules()

    if not initial_missing:
        print("Runtime ready.")
        if initial_optional_missing:
            print("Optional packages missing:", ", ".join(OPTIONAL_MODULES[name] for name in initial_optional_missing))
            print("App fallback face recognizer mode mein chalegi.")
        return 0

    print("Missing packages detected:", ", ".join(REQUIRED_MODULES[name] for name in initial_missing))
    if initial_optional_missing:
        print("Optional packages missing:", ", ".join(OPTIONAL_MODULES[name] for name in initial_optional_missing))
    print("Trying to install them into .vendor312 using this Python:")
    print(sys.executable)
    print("Python version:")
    print(sys.version.split()[0])

    install_result = install_missing_packages()
    final_missing = missing_modules()

    if install_result.returncode == 0 and not final_missing:
        final_optional_missing = missing_optional_modules()
        print("Runtime ready.")
        if final_optional_missing:
            print("Optional packages still missing:", ", ".join(OPTIONAL_MODULES[name] for name in final_optional_missing))
            print("App fallback face recognizer mode mein chalegi.")
        return 0

    print()
    print("Automatic setup could not finish.")
    pip_output = "\n".join(
        line.strip()
        for line in (install_result.stderr or install_result.stdout or "").splitlines()
        if line.strip()
    )
    if "WinError 10013" in pip_output:
        print("Package download is blocked right now. Please allow internet/package access and run again.")
    elif "No matching distribution found" in pip_output or "Could not find a version that satisfies the requirement" in pip_output:
        print("A required package could not be downloaded for the current Python/runtime setup.")
        print("This usually means either internet/package access is blocked, or the current Python version is not supported by one of the packages.")
    if final_missing:
        print("Still missing:", ", ".join(REQUIRED_MODULES[name] for name in final_missing))
    elif pip_output:
        print("Pip reported an unexpected issue while preparing the runtime.")
    if pip_output:
        pip_lines = pip_output.splitlines()
        important_line = next(
            (
                line
                for line in reversed(pip_lines)
                if "ERROR:" in line or "Failed to establish a new connection" in line
            ),
            pip_lines[-1],
        )
        print("Last pip message:")
        print(important_line)
    print("Run this command after allowing internet/package access:")
    print(f"\"{sys.executable}\" -m pip install --target \"{VENDOR_DIR}\" -r \"{REQUIREMENTS_FILE}\"")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
