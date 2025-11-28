"""
ADB DIAGNOSTIC TOOL
Check if ADB is installed and if devices are connected
"""

import os
import subprocess
import shutil
from pathlib import Path


def check_adb_in_path():
    """Check if ADB is in system PATH"""
    print("\n" + "="*70)
    print("1. CHECKING ADB IN SYSTEM PATH")
    print("="*70)
    
    adb_path = shutil.which("adb")
    
    if adb_path:
        print(f"[OK] ADB found in PATH: {adb_path}")
        return adb_path
    else:
        print("[FAIL] ADB not found in system PATH")
        return None


def check_common_adb_locations():
    """Check common ADB installation locations"""
    print("\n" + "="*70)
    print("2. CHECKING COMMON ADB LOCATIONS")
    print("="*70)
    
    common_paths = [
        os.path.expanduser("~\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"),
        "C:\\Android\\sdk\\platform-tools\\adb.exe",
        os.path.expanduser("~\\Android\\Sdk\\platform-tools\\adb.exe"),
        "C:\\Program Files\\Android\\Android Studio\\sdk\\platform-tools\\adb.exe",
    ]
    
    found_paths = []
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"[OK] Found: {path}")
            found_paths.append(path)
        else:
            print(f"[FAIL] Not found: {path}")
    
    return found_paths


def check_adb_version(adb_path):
    """Check ADB version"""
    print("\n" + "="*70)
    print("3️⃣ CHECKING ADB VERSION")
    print("="*70)
    
    try:
        result = subprocess.run(
            [adb_path, "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✅ ADB Version:\n{result.stdout}")
        return True
    except Exception as e:
        print(f"❌ Error checking ADB version: {str(e)}")
        return False


def check_connected_devices(adb_path):
    """Check connected devices"""
    print("\n" + "="*70)
    print("4️⃣ CHECKING CONNECTED DEVICES")
    print("="*70)
    
    try:
        result = subprocess.run(
            [adb_path, "devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print("ADB Devices Output:")
        print(result.stdout)
        
        # Parse devices
        devices_list = []
        for line in result.stdout.split('\n')[1:]:
            if line.strip() and 'device' in line and 'List' not in line:
                device_info = line.split()[0]
                if device_info:
                    devices_list.append(device_info)
        
        if devices_list:
            print(f"\n✅ Found {len(devices_list)} connected device(s):")
            for device in devices_list:
                print(f"   - {device}")
            return devices_list
        else:
            print("\n❌ No devices detected")
            print("   Make sure:")
            print("   1. Device is connected via USB")
            print("   2. USB debugging is enabled on device")
            print("   3. Device is unlocked")
            return []
    
    except Exception as e:
        print(f"❌ Error checking devices: {str(e)}")
        return []


def check_usb_drivers():
    """Check USB driver status"""
    print("\n" + "="*70)
    print("5️⃣ CHECKING USB DRIVERS")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["adb", "devices", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print("Detailed Device Info:")
        print(result.stdout)
        
        if "offline" in result.stdout:
            print("\n⚠️ Device is OFFLINE - USB drivers may not be installed")
            print("   Solution: Install Android USB drivers")
        elif "unauthorized" in result.stdout:
            print("\n⚠️ Device is UNAUTHORIZED")
            print("   Solution: Tap 'Allow' on device when prompted")
        elif "device" in result.stdout:
            print("\n✅ Device is properly connected")
        
    except Exception as e:
        print(f"⚠️ Could not check detailed device info: {str(e)}")


def main():
    """Run all diagnostics"""
    
    print("\n")
    print("=" * 70)
    print("ADB DIAGNOSTIC TOOL".center(70))
    print("=" * 70)
    
    # Step 1: Check PATH
    adb_path = check_adb_in_path()
    
    # Step 2: Check common locations
    found_paths = check_common_adb_locations()
    
    # Use found ADB or first common path
    if adb_path:
        adb_to_use = adb_path
    elif found_paths:
        adb_to_use = found_paths[0]
    else:
        adb_to_use = None
    
    if not adb_to_use:
        print("\n" + "="*70)
        print("❌ ADB NOT FOUND")
        print("="*70)
        print("\n📥 SOLUTION: Install Android SDK Platform Tools")
        print("\nOption 1: Download from Google")
        print("   1. Go to: https://developer.android.com/tools/releases/platform-tools")
        print("   2. Download platform-tools-latest-windows.zip")
        print("   3. Extract to: C:\\Android\\sdk\\platform-tools\\")
        print("   4. Add to PATH or use full path")
        print("\nOption 2: Use Android Studio")
        print("   1. Install Android Studio")
        print("   2. ADB will be at: C:\\Users\\[username]\\AppData\\Local\\Android\\Sdk\\platform-tools\\")
        print("\nOption 3: Use Desktop Tool (doesn't need web app)")
        print("   1. Run: python desktop_extraction_tool.py")
        print("   2. Desktop tool can detect devices without ADB in PATH")
        return
    
    print(f"\n✅ Using ADB: {adb_to_use}")
    
    # Step 3: Check version
    check_adb_version(adb_to_use)
    
    # Step 4: Check devices
    devices = check_connected_devices(adb_to_use)
    
    # Step 5: Check USB drivers
    check_usb_drivers()
    
    # Summary
    print("\n" + "="*70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if devices:
        print(f"✅ SUCCESS: {len(devices)} device(s) detected")
        print("   The web app should now detect your device!")
    else:
        print("❌ NO DEVICES DETECTED")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check USB cable connection")
        print("   2. Enable USB debugging on device")
        print("   3. Unlock device")
        print("   4. Tap 'Allow' if prompted on device")
        print("   5. Try: adb kill-server && adb start-server")
        print("   6. Reconnect device")
        print("\n💡 ALTERNATIVE: Use Desktop Tool")
        print("   python desktop_extraction_tool.py")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
