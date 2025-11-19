"""Structured audit trail for consent portal approvals."""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class ConsentAuditTrail:
    """Structured audit trail for consent portal approvals."""
    
    AUDIT_FILE = Path('audit/consent_portal/audit_trail.json')
    
    @classmethod
    def initialize(cls):
        """Create audit file if needed."""
        cls.AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls.AUDIT_FILE.exists():
            cls.AUDIT_FILE.write_text(json.dumps([], indent=2))
    
    @classmethod
    def record_approval(cls,
                       case_id: str,
                       decision: str,
                       nominee_name: str,
                       device_id: str,
                       purpose: str = "Not specified",
                       nominee_phone: Optional[str] = None,
                       nominee_email: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> bool:
        """Record approval decision to audit trail."""
        try:
            cls.initialize()
            
            # Read existing trail
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            # Create new entry
            entry = {
                'id': len(trail) + 1,
                'timestamp': datetime.now().isoformat(),
                'case_id': case_id,
                'decision': decision,
                'nominee_name': nominee_name,
                'device_id': device_id,
                'purpose': purpose,
                'nominee_phone': nominee_phone,
                'nominee_email': nominee_email,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'status': 'recorded'
            }
            
            trail.append(entry)
            
            # Write back
            cls.AUDIT_FILE.write_text(json.dumps(trail, indent=2))
            return True
        except Exception as e:
            print(f"Failed to record audit trail: {e}")
            return False
    
    @classmethod
    def get_audit_trail(cls, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit trail, optionally filtered by case_id."""
        try:
            cls.initialize()
            trail = json.loads(cls.AUDIT_FILE.read_text())
            
            if case_id:
                return [entry for entry in trail if entry['case_id'] == case_id]
            return trail
        except Exception:
            return []
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get audit trail statistics."""
        trail = cls.get_audit_trail()
        
        return {
            'total_records': len(trail),
            'approvals': len([e for e in trail if e['decision'] == 'approved']),
            'denials': len([e for e in trail if e['decision'] == 'denied']),
            'cases': len(set(e['case_id'] for e in trail)),
            'first_record': trail[0]['timestamp'] if trail else None,
            'last_record': trail[-1]['timestamp'] if trail else None
        }
    
    @classmethod
    def export_audit_trail(cls, case_id: Optional[str] = None) -> str:
        """Export audit trail as JSON string."""
        trail = cls.get_audit_trail(case_id)
        return json.dumps(trail, indent=2)


__all__ = ["ConsentAuditTrail"]
