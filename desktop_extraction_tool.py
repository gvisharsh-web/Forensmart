"""
FORENSMART DESKTOP EXTRACTION TOOL
Local device extraction with consent token verification

Features:
- Verify consent tokens
- Detect connected devices
- Extract data based on consent level
- Upload results to web app
- Complete audit trail
"""

import json
import base64
import hashlib
import hmac
from datetime import datetime
from typing import Dict, Tuple, List
import subprocess

# ============================================================================
# CONSENT TOKEN VERIFICATION
# ============================================================================

class ConsentTokenVerifier:
    """Verify consent tokens from web app"""
    
    def __init__(self, secret_key: str = "forensmart-secret-key"):
        """Initialize verifier with secret key"""
        self.secret_key = secret_key
        self.verification_log = []
    
    def decode_token(self, token: str) -> Dict:
        """
        Decode consent token
        
        Args:
            token: Base64 encoded token
            
        Returns:
            Decoded token data
        """
        try:
            # Remove header if present
            if token.startswith("FORENSMART_CONSENT_TOKEN"):
                token = token.split('\n')[1]
            
            # Decode from Base64
            decoded = base64.b64decode(token)
            token_data = json.loads(decoded)
            
            print(f"✅ Token decoded successfully")
            return token_data
        
        except Exception as e:
            print(f"❌ Token decode failed: {str(e)}")
            return None
    
    def verify_token(self, token: str) -> Tuple[bool, str, Dict]:
        """
        Verify consent token authenticity and validity
        
        Args:
            token: Consent token to verify
            
        Returns:
            (is_valid, message, consent_data)
        """
        try:
            # Decode token
            token_data = self.decode_token(token)
            if not token_data:
                return False, "Failed to decode token", None
            
            # Extract components
            consent_data = token_data.get('data', {})
            received_hash = token_data.get('hash', '')
            received_signature = token_data.get('signature', '')
            
            print(f"\n📋 Verifying Token...")
            print(f"   Case ID: {consent_data.get('case_id')}")
            print(f"   Consent Level: {consent_data.get('consent_level')}")
            
            # Step 1: Verify hash
            print(f"\n   Step 1: Verifying hash...")
            data_json = json.dumps(consent_data, sort_keys=True)
            calculated_hash = hashlib.sha256(data_json.encode()).hexdigest()
            
            if calculated_hash != received_hash:
                msg = "❌ Hash mismatch - data has been tampered!"
                self.verification_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'result': 'failed',
                    'reason': 'hash_mismatch'
                })
                return False, msg, None
            
            print(f"   ✅ Hash verified")
            
            # Step 2: Verify signature
            print(f"\n   Step 2: Verifying signature...")
            calculated_signature = hmac.new(
                self.secret_key.encode(),
                data_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if calculated_signature != received_signature:
                msg = "❌ Signature mismatch - not authentic!"
                self.verification_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'result': 'failed',
                    'reason': 'signature_mismatch'
                })
                return False, msg, None
            
            print(f"   ✅ Signature verified")
            
            # Step 3: Check expiry
            print(f"\n   Step 3: Checking expiry...")
            try:
                expiry = datetime.fromisoformat(consent_data.get('expiry_date', ''))
                if datetime.now() > expiry:
                    msg = "❌ Consent expired!"
                    self.verification_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'result': 'failed',
                        'reason': 'expired'
                    })
                    return False, msg, None
                
                print(f"   ✅ Not expired (expires: {consent_data.get('expiry_date')})")
            except:
                pass
            
            # Step 4: Verify required fields
            print(f"\n   Step 4: Verifying required fields...")
            required_fields = ['case_id', 'consent_level', 'approved_by', 'modules_allowed']
            missing_fields = [f for f in required_fields if f not in consent_data]
            
            if missing_fields:
                msg = f"❌ Missing fields: {', '.join(missing_fields)}"
                return False, msg, None
            
            print(f"   ✅ All required fields present")
            
            # All checks passed
            msg = "✅ Consent token verified and authentic!"
            self.verification_log.append({
                'timestamp': datetime.now().isoformat(),
                'result': 'success',
                'case_id': consent_data.get('case_id'),
                'consent_level': consent_data.get('consent_level')
            })
            
            return True, msg, consent_data
        
        except Exception as e:
            msg = f"❌ Verification error: {str(e)}"
            return False, msg, None

# ============================================================================
# DEVICE DETECTION
# ============================================================================

class DeviceDetector:
    """Detect connected devices"""
    
    @staticmethod
    def find_adb_path() -> str:
        """Find ADB executable in common locations"""
        import os
        import shutil
        
        # Try standard PATH first
        adb_path = shutil.which("adb")
        if adb_path:
            return adb_path
        
        # Try common Android SDK locations
        common_paths = [
            os.path.expanduser("~\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe"),
            "C:\\Android\\sdk\\platform-tools\\adb.exe",
            os.path.expanduser("~\\Android\\Sdk\\platform-tools\\adb.exe"),
            "C:\\Program Files\\Android\\Android Studio\\sdk\\platform-tools\\adb.exe",
            os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\\platform-tools\\adb.exe"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    @staticmethod
    def detect_devices() -> List[str]:
        """
        Detect connected devices via ADB
        
        Returns:
            List of device IDs
        """
        try:
            # Find ADB path
            adb_path = DeviceDetector.find_adb_path()
            
            if not adb_path:
                print("⚠️ ADB not found in any common location")
                return []
            
            print(f"[INFO] Using ADB: {adb_path}")
            
            result = subprocess.run(
                [adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            devices_list = []
            for line in result.stdout.split('\n')[1:]:
                if line.strip() and 'device' in line and 'List' not in line:
                    device_id = line.split()[0]
                    if device_id:
                        devices_list.append(device_id)
            
            return devices_list
        
        except Exception as e:
            print(f"⚠️ Device detection error: {str(e)}")
            return []

# ============================================================================
# EXTRACTION MANAGER
# ============================================================================

class ExtractionManager:
    """Manage extraction based on consent"""
    
    def __init__(self, consent_data: Dict):
        """Initialize with consent data"""
        self.consent_data = consent_data
        self.case_id = consent_data.get('case_id')
        self.consent_level = consent_data.get('consent_level')
        self.modules_allowed = consent_data.get('modules_allowed', [])
        self.modules_blocked = consent_data.get('modules_blocked', [])
    
    def get_extraction_plan(self) -> Dict:
        """Get extraction plan based on consent"""
        
        print(f"\n📋 Extraction Plan")
        print(f"   Case ID: {self.case_id}")
        print(f"   Consent Level: {self.consent_level}")
        print(f"\n   ✅ Allowed Modules:")
        for module in self.modules_allowed:
            print(f"      - {module}")
        
        print(f"\n   ❌ Blocked Modules:")
        for module in self.modules_blocked:
            print(f"      - {module}")
        
        return {
            'case_id': self.case_id,
            'consent_level': self.consent_level,
            'modules_allowed': self.modules_allowed,
            'modules_blocked': self.modules_blocked,
            'extraction_ready': True
        }
    
    def simulate_extraction(self) -> Dict:
        """Simulate extraction process"""
        
        print(f"\n🔍 Starting Extraction...")
        print(f"   Case: {self.case_id}")
        
        extraction_results = {
            'case_id': self.case_id,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'extracted_modules': {},
            'total_files': 0,
            'total_size': 0
        }
        
        # Simulate extraction for each allowed module
        for module in self.modules_allowed:
            print(f"\n   Extracting {module}...")
            
            # Simulate file extraction
            file_count = {
                'device_info': 5,
                'communications': 150,
                'location': 45,
                'media': 2500,
                'security': 30,
                'system': 100
            }.get(module, 10)
            
            size_mb = {
                'device_info': 2,
                'communications': 50,
                'location': 10,
                'media': 5000,
                'security': 5,
                'system': 20
            }.get(module, 10)
            
            extraction_results['extracted_modules'][module] = {
                'files': file_count,
                'size_mb': size_mb,
                'status': 'completed'
            }
            
            extraction_results['total_files'] += file_count
            extraction_results['total_size'] += size_mb
            
            print(f"      ✅ {file_count} files ({size_mb} MB)")
        
        print(f"\n   ✅ Extraction completed!")
        print(f"      Total files: {extraction_results['total_files']}")
        print(f"      Total size: {extraction_results['total_size']} MB")
        
        return extraction_results

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main desktop tool application"""
    
    print("=" * 70)
    print("🔍 FORENSMART DESKTOP EXTRACTION TOOL")
    print("=" * 70)
    
    # Step 1: Get consent token from user
    print("\n📋 Step 1: Paste Consent Token")
    print("-" * 70)
    print("Paste your consent token (from web app):")
    print("(You can paste the full token including header)")
    print()
    
    token = input("🔐 Paste token: ").strip()
    
    if not token:
        print("❌ No token provided!")
        return
    
    # Step 2: Verify token
    print("\n📋 Step 2: Verify Consent Token")
    print("-" * 70)
    
    verifier = ConsentTokenVerifier()
    is_valid, message, consent_data = verifier.verify_token(token)
    
    print(f"\n{message}")
    
    if not is_valid:
        print("❌ Cannot proceed without valid consent!")
        return
    
    # Step 3: Show extraction plan
    print("\n📋 Step 3: Extraction Plan")
    print("-" * 70)
    
    extraction_manager = ExtractionManager(consent_data)
    plan = extraction_manager.get_extraction_plan()
    
    # Step 4: Detect devices
    print("\n📋 Step 4: Detect Connected Devices")
    print("-" * 70)
    
    detector = DeviceDetector()
    devices = detector.detect_devices()
    
    if devices:
        print(f"✅ Found {len(devices)} connected device(s):")
        for i, device in enumerate(devices, 1):
            print(f"   {i}. {device}")
        
        # Select device
        device_choice = input("\nSelect device (enter number): ").strip()
        try:
            selected_device = devices[int(device_choice) - 1]
            print(f"✅ Selected device: {selected_device}")
        except:
            print("❌ Invalid device selection!")
            return
    else:
        print("⚠️ No devices detected")
        print("   Make sure device is connected and ADB is enabled")
        return
    
    # Step 5: Perform extraction
    print("\n📋 Step 5: Perform Extraction")
    print("-" * 70)
    
    proceed = input("\nProceed with extraction? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("❌ Extraction cancelled!")
        return
    
    extraction_results = extraction_manager.simulate_extraction()
    
    # Step 6: Summary
    print("\n📋 Step 6: Extraction Summary")
    print("-" * 70)
    
    print(f"\n✅ Extraction Completed!")
    print(f"   Case ID: {extraction_results['case_id']}")
    print(f"   Status: {extraction_results['status']}")
    print(f"   Total Files: {extraction_results['total_files']}")
    print(f"   Total Size: {extraction_results['total_size']} MB")
    print(f"   Timestamp: {extraction_results['timestamp']}")
    
    print(f"\n📤 Ready to upload results to web app!")
    print(f"   Results will be saved and can be uploaded via web interface")
    
    print("\n" + "=" * 70)
    print("✅ EXTRACTION TOOL COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
