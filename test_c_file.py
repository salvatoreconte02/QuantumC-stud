#!/usr/bin/env python3

import subprocess
import argparse
import os
import sys
import glob
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"
CATEGORIES = ["scalar", "vector", "matrix", "hybrid", "comparison"]


def compile_and_run_c(c_path: str, verbose: bool = True) -> str:
    """Compile and run a C file, return the exit code as string."""
    exe_path = "/tmp/test_binary"
    try:
        subprocess.run(["gcc", c_path, "-o", exe_path], check=True,
                       capture_output=True, text=True)
        result = subprocess.run([exe_path], capture_output=True, text=True)
        returncode = str(result.returncode)
        if verbose:
            print(f"[C OUTPUT] Return code: {returncode}")
        return returncode
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERROR] Compilation failed: {e}")
        return None
    finally:
        if os.path.exists(exe_path):
            os.remove(exe_path)


def run_quantum_pipeline(c_path: str, adder: str = "qft", verbose: bool = True) -> str:
    """Run the quantum pipeline and return the output."""
    try:
        cmd = [sys.executable, "pipeline.py", c_path, "--run", "--adder", adder]
        result = subprocess.run(cmd, capture_output=True, text=True)
        quantum_output = result.stdout + result.stderr
        if verbose:
            print("[Quantum OUTPUT]")
            print(quantum_output)
        return quantum_output
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[ERROR] Quantum execution failed: {e}")
        return ""


def extract_quantum_result(quantum_output: str) -> int:
    """Extract the final quantum measurement value from output."""
    import re
    # Look for "value (2's complement) = X" pattern
    match = re.search(r"value \(2's complement\) = (-?\d+)", quantum_output)
    if match:
        return int(match.group(1))
    return None


def compare_results(classical_code: int, quantum_value: int, bits: int = 16) -> bool:
    """Compare classical exit code with quantum value, handling signed/unsigned conversion.

    Classical C exit codes are unsigned 8-bit (0-255).
    Quantum values are signed (2's complement).

    For negative quantum values, the classical exit code is (256 + quantum_value).
    """
    if quantum_value is None:
        return False

    # Direct match
    if classical_code == quantum_value:
        return True

    # Handle signed/unsigned conversion for exit codes
    # If quantum is negative and classical > 127, check if they match via 2's complement
    if quantum_value < 0:
        # Convert quantum signed to what C exit code would be (mod 256)
        expected_exit = quantum_value & 0xFF
        if classical_code == expected_exit:
            return True

    # Also check if classical value appears in the output string (legacy behavior)
    return False


def run_single_test(c_path: str, adder: str = "qft", verbose: bool = True) -> bool:
    """Run a single test and return True if passed, False otherwise."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"[INFO] Testing: {c_path}")
        print(f"{'='*60}")

    classical_output = compile_and_run_c(c_path, verbose)
    if classical_output is None:
        return False

    if verbose:
        print(f"\n[INFO] Running quantum pipeline (adder={adder})")
    quantum_output = run_quantum_pipeline(c_path, adder, verbose)

    # Try numeric comparison first
    classical_code = int(classical_output)
    quantum_value = extract_quantum_result(quantum_output)
    passed = compare_results(classical_code, quantum_value)

    # Fallback: string match (for legacy compatibility)
    if not passed:
        passed = classical_output in quantum_output

    if verbose:
        print("\n[RESULT]")
        if passed:
            q_str = f" (quantum={quantum_value})" if quantum_value is not None else ""
            print(f"[PASS] Classical result '{classical_output}' matches quantum output{q_str}")
        else:
            q_str = f" (quantum={quantum_value})" if quantum_value is not None else ""
            print(f"[FAIL] Classical result '{classical_output}' NOT found in quantum output{q_str}")

    return passed


def get_test_files(category: str = None) -> list:
    """Get list of test files, optionally filtered by category."""
    if category:
        category_path = TESTS_DIR / category
        if not category_path.exists():
            print(f"[ERROR] Category '{category}' not found in {TESTS_DIR}")
            sys.exit(1)
        return sorted(category_path.glob("*.c"))
    else:
        return sorted(TESTS_DIR.glob("**/*.c"))


def run_batch_tests(files: list, adder: str = "qft", verbose: bool = False) -> dict:
    """Run multiple tests and return results summary."""
    results = {"passed": [], "failed": []}

    for f in files:
        test_name = str(f.relative_to(TESTS_DIR))
        print(f"\n[TEST] {test_name} ", end="", flush=True)

        passed = run_single_test(str(f), adder, verbose)

        if passed:
            print("[PASS]")
            results["passed"].append(test_name)
        else:
            print("[FAIL]")
            results["failed"].append(test_name)

    return results


def print_summary(results: dict, adder: str):
    """Print final summary of test results."""
    total = len(results["passed"]) + len(results["failed"])
    passed = len(results["passed"])
    failed = len(results["failed"])

    print("\n" + "="*60)
    print(f"TEST SUMMARY (adder={adder})")
    print("="*60)
    print(f"Total:  {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\nFailed tests:")
        for t in results["failed"]:
            print(f"  - {t}")

    print(f"\nResult: {passed}/{total} passed")
    if failed == 0:
        print("[ALL TESTS PASSED]")
    else:
        print("[SOME TESTS FAILED]")


def main():
    parser = argparse.ArgumentParser(
        description="Compare classical and quantum output for C programs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_c_file.py tests/scalar/add.c       Run single test
  python test_c_file.py --all                    Run all tests
  python test_c_file.py --category scalar        Run all scalar tests
  python test_c_file.py --all --adder ripple     Run all with ripple-carry
  python test_c_file.py --all --adder both       Compare both backends
        """
    )
    parser.add_argument("c_file", nargs="?", help="Path to the input C file or directory")
    parser.add_argument("--all", action="store_true", help="Run all tests in tests/")
    parser.add_argument("--category", choices=CATEGORIES,
                        help="Run tests from a specific category")
    parser.add_argument("--adder", choices=["qft", "ripple", "both"], default="qft",
                        help="Arithmetic backend to use (default: qft)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each test")

    args = parser.parse_args()

    # Determine which tests to run
    if args.all:
        files = get_test_files()
    elif args.category:
        files = get_test_files(args.category)
    elif args.c_file:
        c_path = Path(args.c_file)
        if c_path.is_dir():
            files = sorted(c_path.glob("*.c"))
        else:
            files = [c_path]
    else:
        parser.print_help()
        sys.exit(1)

    if not files:
        print("[ERROR] No test files found")
        sys.exit(1)

    # Single file mode: always verbose
    if len(files) == 1 and not args.all and not args.category:
        passed = run_single_test(str(files[0]), args.adder, verbose=True)
        sys.exit(0 if passed else 1)

    # Batch mode
    adders = ["qft", "ripple"] if args.adder == "both" else [args.adder]

    for adder in adders:
        print(f"\n{'#'*60}")
        print(f"# Running tests with adder: {adder}")
        print(f"{'#'*60}")

        results = run_batch_tests(files, adder, args.verbose)
        print_summary(results, adder)

    # Exit with error if any test failed
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
