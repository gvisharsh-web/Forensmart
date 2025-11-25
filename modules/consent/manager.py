"""
Consent Manager Helper
Provides convenience functions for consent management
"""

from modules.consent.models import ConsentManager

# Global instance
_consent_manager_instance = None

def get_consent_manager() -> ConsentManager:
    """Get or create the global ConsentManager instance"""
    global _consent_manager_instance
    
    if _consent_manager_instance is None:
        _consent_manager_instance = ConsentManager()
    
    return _consent_manager_instance

# Re-export ConsentAuditTrail if available
try:
    from modules.consent.portal import ConsentAuditTrail
except ImportError:
    ConsentAuditTrail = None
