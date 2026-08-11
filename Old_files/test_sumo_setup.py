"""
Diagnostic script to verify SUMO installation and environment setup.
Run this BEFORE attempting to start the full environment.
"""

import os
import sys
from pathlib import Path
import subprocess


def check_sumo_home():
    """Check if SUMO_HOME environment variable is set."""
    print("=" * 70)
    print("1. Checking SUMO_HOME environment variable...")
    print("=" * 70)
    
    #sumo_home = os.getenv("SUMO_HOME")
    sumo_home=r"C:\Program Files (x86)\Eclipse\Sumo"
    if sumo_home:
        print(f"✓ SUMO_HOME is set to: {sumo_home}")
        sumo_path = Path(sumo_home)
        if sumo_path.is_dir():
            print(f"✓ SUMO_HOME directory exists")
            return sumo_home
        else:
            print(f"✗ SUMO_HOME directory NOT FOUND: {sumo_home}")
            return None
    else:
        print(f"✗ SUMO_HOME environment variable is NOT set")
        print(f"  Please set it to your SUMO installation directory")
        return None


def check_sumo_binaries(sumo_home):
    """Check if SUMO executables exist."""
    print("\n" + "=" * 70)
    print("2. Checking SUMO binary executables...")
    print("=" * 70)
    
    if not sumo_home:
        print("✗ Cannot check binaries (SUMO_HOME not set)")
        return False
    
    bin_dir = Path(sumo_home) / "bin"
    print(f"Looking in: {bin_dir}")
    
    if not bin_dir.is_dir():
        print(f"✗ bin directory not found")
        return False
    
    print(f"✓ bin directory found")
    
    sumo_exe = bin_dir / "sumo.exe"
    sumo_gui_exe = bin_dir / "sumo-gui.exe"
    
    sumo_found = sumo_exe.is_file()
    sumo_gui_found = sumo_gui_exe.is_file()
    
    print(f"  sumo.exe: {'✓' if sumo_found else '✗'} {sumo_exe}")
    print(f"  sumo-gui.exe: {'✓' if sumo_gui_found else '✗'} {sumo_gui_exe}")
    
    return sumo_found or sumo_gui_found


def check_sumo_version(sumo_home):
    """Check SUMO version by running sumo --version."""
    print("\n" + "=" * 70)
    print("3. Checking SUMO version...")
    print("=" * 70)
    
    if not sumo_home:
        print("✗ Cannot check version (SUMO_HOME not set)")
        return False
    
    sumo_exe = Path(sumo_home) / "bin" / "sumo.exe"
    
    try:
        result = subprocess.run(
            [str(sumo_exe), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ SUMO version check successful:")
            print(f"  {result.stdout.strip()}")
            return True
        else:
            print(f"✗ SUMO version check failed")
            print(f"  stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running sumo --version: {e}")
        return False


def check_traci():
    """Check if TraCI/SUMO Python bindings are installed."""
    print("\n" + "=" * 70)
    print("4. Checking SUMO Python bindings (TraCI)...")
    print("=" * 70)
    
    try:
        import traci
        print(f"✓ traci module imported successfully")
        print(f"  traci module location: {traci.__file__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import traci: {e}")
        print(f"  Install with: pip install traci")
        return False


def check_gymnasium():
    """Check if gymnasium is installed."""
    print("\n" + "=" * 70)
    print("5. Checking gymnasium...")
    print("=" * 70)
    
    try:
        import gymnasium as gym
        print(f"✓ gymnasium module imported successfully")
        print(f"  gymnasium module location: {gym.__file__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import gymnasium: {e}")
        print(f"  Install with: pip install gymnasium")
        return False


def check_network_files():
    """Check if your network files exist."""
    print("\n" + "=" * 70)
    print("6. Checking SUMO network files...")
    print("=" * 70)
    
    BASE_DIR = r"D:\Users\NK\Resume\Projects_to_Github\Multi-Agent-SUMO_DQN"
    MAPS_DIR = os.path.join(BASE_DIR, "Maps")
    NET_FILE = os.path.join(MAPS_DIR, "network2.net.xml")
    ROUTE_FILE = os.path.join(MAPS_DIR, "routes2.rou.xml")
    
    net_exists = Path(NET_FILE).is_file()
    route_exists = Path(ROUTE_FILE).is_file()
    
    print(f"Network file (.net.xml): {'✓' if net_exists else '✗'} {NET_FILE}")
    print(f"Routes file (.rou.xml):  {'✓' if route_exists else '✗'} {ROUTE_FILE}")
    
    return net_exists and route_exists


def main():
    print("\n")
    print("█" * 70)
    print("SUMO Setup Diagnostic")
    print("█" * 70)
    print()
    
    results = {}
    
    # Run all checks
    sumo_home = check_sumo_home()
    results["sumo_home"] = sumo_home is not None
    
    if sumo_home:
        results["sumo_binaries"] = check_sumo_binaries(sumo_home)
        results["sumo_version"] = check_sumo_version(sumo_home)
    else:
        results["sumo_binaries"] = False
        results["sumo_version"] = False
    
    results["traci"] = check_traci()
    results["gymnasium"] = check_gymnasium()
    results["network_files"] = check_network_files()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check}")
    
    print()
    if all_passed:
        print("✓✓✓ All checks passed! You're ready to run the SUMO environment.")
        print()
        print("Next steps:")
        print("  1. Copy sumo_env_fixed.py to your project")
        print("  2. Update your Python script to use sumo_env_fixed.py")
        print("  3. Run your training/simulation script")
    else:
        print("✗✗✗ Some checks failed. Fix the issues above and try again.")
        print()
        print("Common solutions:")
        print("  - If SUMO_HOME is not set, open Settings > Environment Variables")
        print("    and add a new User variable: SUMO_HOME = C:\\Program Files (x86)\\Eclipse\\Sumo")
        print("  - If traci is missing: pip install traci")
        print("  - If gymnasium is missing: pip install gymnasium")
    
    print()


if __name__ == "__main__":
    main()
