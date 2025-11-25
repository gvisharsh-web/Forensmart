"""
Phase 4: Consent Level Hierarchy Tests
======================================

Unit tests for the immutable consent level system and module requirements.

Tests:
1. Consent level immutability
2. Module consent checking
3. Consent hierarchy (NONE < BASIC < STANDARD < LEGAL < FULL)
4. Error messages clarity
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.consent.models import ConsentManager, ConsentLevel, ConsentSession
from modules.extraction.orchestrator import DataExtractionOrchestrator, MODULE_MIN_LEVELS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConsentLevelImmutability:
    """Test consent level immutability"""
    
    def test_set_consent_level_immutable(self):
        """Test setting consent level with immutability lock"""
        cm = ConsentManager()
        case_id = "TEST_CASE_001"
        
        # Create a session with required parameters
        session = ConsentSession(case_id, device_id="TEST_DEVICE", level=ConsentLevel.NONE)
        cm.sessions[case_id] = session
        
        # Set consent level immutable
        result = cm.set_consent_level_immutable(
            case_id, 
            ConsentLevel.LEGAL, 
            reason="Test immutability"
        )
        
        assert result == True, "Should successfully set consent level"
        assert cm.is_consent_level_locked(case_id) == True, "Should be locked"
        assert cm.get_consent_level(case_id) == ConsentLevel.LEGAL, "Should be LEGAL"
        
        logger.info("[PASS] test_set_consent_level_immutable PASSED")
    
    def test_consent_level_cannot_be_changed_when_locked(self):
        """Test that locked consent level cannot be changed"""
        cm = ConsentManager()
        case_id = "TEST_CASE_002"
        
        # Create a session with required parameters
        session = ConsentSession(case_id, device_id="TEST_DEVICE", level=ConsentLevel.NONE)
        cm.sessions[case_id] = session
        
        # Set consent level immutable
        cm.set_consent_level_immutable(case_id, ConsentLevel.LEGAL)
        
        # Try to change it
        result = cm.set_consent_level_immutable(case_id, ConsentLevel.STANDARD)
        
        assert result == False, "Should not allow changing locked consent level"
        assert cm.get_consent_level(case_id) == ConsentLevel.LEGAL, "Should still be LEGAL"
        
        logger.info("✅ test_consent_level_cannot_be_changed_when_locked PASSED")
    
    def test_get_consent_level_info(self):
        """Test getting detailed consent level information"""
        cm = ConsentManager()
        case_id = "TEST_CASE_003"
        
        # Create a session with required parameters
        session = ConsentSession(case_id, device_id="TEST_DEVICE", level=ConsentLevel.NONE)
        cm.sessions[case_id] = session
        
        # Set consent level
        cm.set_consent_level_immutable(case_id, ConsentLevel.LEGAL)
        
        # Get info
        info = cm.get_consent_level_info(case_id)
        
        assert info['level'] == 'LEGAL', "Level should be LEGAL"
        assert info['level_value'] == 4, "Level value should be 4"
        assert info['locked'] == True, "Should be locked"
        assert info['scope'] is not None, "Should have scope"
        
        logger.info("✅ test_get_consent_level_info PASSED")


class TestModuleConsentRequirements:
    """Test module consent requirements"""
    
    def test_module_min_levels_defined(self):
        """Test that all modules have minimum levels defined"""
        expected_modules = ['device_info', 'communications', 'location', 'security', 'media', 'system']
        
        for module in expected_modules:
            assert module in MODULE_MIN_LEVELS, f"Module {module} not in MODULE_MIN_LEVELS"
        
        logger.info("✅ test_module_min_levels_defined PASSED")
    
    def test_communications_requires_legal(self):
        """Test that communications module requires LEGAL consent"""
        assert MODULE_MIN_LEVELS['communications'] == ConsentLevel.LEGAL, \
            "Communications should require LEGAL consent"
        
        logger.info("✅ test_communications_requires_legal PASSED")
    
    def test_location_requires_standard(self):
        """Test that location module requires STANDARD consent"""
        assert MODULE_MIN_LEVELS['location'] == ConsentLevel.STANDARD, \
            "Location should require STANDARD consent"
        
        logger.info("✅ test_location_requires_standard PASSED")
    
    def test_check_module_consent_legal_passes_communications(self):
        """Test that LEGAL consent passes communications check"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.LEGAL)
        
        assert allowed == True, "LEGAL should pass communications check"
        assert '✅' in message, "Message should indicate success"
        
        logger.info("✅ test_check_module_consent_legal_passes_communications PASSED")
    
    def test_check_module_consent_standard_fails_communications(self):
        """Test that STANDARD consent fails communications check"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.STANDARD)
        
        assert allowed == False, "STANDARD should fail communications check"
        assert '❌' in message, "Message should indicate failure"
        assert 'LEGAL' in message, "Message should mention LEGAL requirement"
        assert 'STANDARD' in message, "Message should mention current level"
        
        logger.info("✅ test_check_module_consent_standard_fails_communications PASSED")
    
    def test_check_module_consent_full_passes_communications(self):
        """Test that FULL consent passes communications check"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.FULL)
        
        assert allowed == True, "FULL should pass communications check"
        
        logger.info("✅ test_check_module_consent_full_passes_communications PASSED")


class TestConsentHierarchy:
    """Test consent level hierarchy (NONE < BASIC < STANDARD < LEGAL < FULL)"""
    
    def test_consent_hierarchy_values(self):
        """Test that consent levels have correct numeric values"""
        assert ConsentLevel.NONE.value == 0, "NONE should be 0"
        assert ConsentLevel.BASIC.value == 1, "BASIC should be 1"
        assert ConsentLevel.STANDARD.value == 2, "STANDARD should be 2"
        assert ConsentLevel.LEGAL.value == 3, "LEGAL should be 3"
        assert ConsentLevel.FULL.value == 4, "FULL should be 4"
        
        logger.info("✅ test_consent_hierarchy_values PASSED")
    
    def test_numeric_comparison_works(self):
        """Test that numeric comparison works correctly"""
        orchestrator = DataExtractionOrchestrator()
        
        # LEGAL (3) >= LEGAL (3) = True
        allowed, _ = orchestrator.check_module_consent('communications', ConsentLevel.LEGAL)
        assert allowed == True
        
        # STANDARD (2) >= LEGAL (3) = False
        allowed, _ = orchestrator.check_module_consent('communications', ConsentLevel.STANDARD)
        assert allowed == False
        
        # FULL (4) >= LEGAL (3) = True
        allowed, _ = orchestrator.check_module_consent('communications', ConsentLevel.FULL)
        assert allowed == True
        
        logger.info("✅ test_numeric_comparison_works PASSED")


class TestErrorMessageClarity:
    """Test that error messages are clear and helpful"""
    
    def test_error_message_includes_required_level(self):
        """Test that error message includes required level"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.STANDARD)
        
        assert 'LEGAL' in message, "Should mention LEGAL requirement"
        assert 'communications' in message.lower(), "Should mention module name"
        
        logger.info("✅ test_error_message_includes_required_level PASSED")
    
    def test_error_message_includes_current_level(self):
        """Test that error message includes current level"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.STANDARD)
        
        assert 'STANDARD' in message, "Should mention current level"
        
        logger.info("✅ test_error_message_includes_current_level PASSED")
    
    def test_success_message_includes_both_levels(self):
        """Test that success message includes both levels"""
        orchestrator = DataExtractionOrchestrator()
        allowed, message = orchestrator.check_module_consent('communications', ConsentLevel.LEGAL)
        
        assert 'LEGAL' in message, "Should mention required level"
        assert '✅' in message, "Should include success indicator"
        
        logger.info("✅ test_success_message_includes_both_levels PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("PHASE 4: CONSENT LEVEL HIERARCHY TESTS")
    print("="*70 + "\n")
    
    test_classes = [
        TestConsentLevelImmutability,
        TestModuleConsentRequirements,
        TestConsentHierarchy,
        TestErrorMessageClarity
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n[TEST] Running {test_class.__name__}...")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    passed_tests += 1
                except AssertionError as e:
                    failed_tests += 1
                    logger.error(f"[FAIL] {method_name} FAILED: {e}")
                except Exception as e:
                    failed_tests += 1
                    logger.error(f"[ERROR] {method_name} ERROR: {e}")
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed_tests}/{total_tests} passed, {failed_tests} failed")
    print("="*70 + "\n")
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
