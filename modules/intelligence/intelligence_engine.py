"""
INTELLIGENCE ENGINE - Pattern analysis, threat detection, and risk assessment

Integrates with:
- Database Module (for data storage)
- API Module (for external data)
- Report Generation (for reporting)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from modules.shared.database import DatabaseManager
from modules.shared.api import APIClient

logger = logging.getLogger(__name__)

# ============================================================================
# INTELLIGENCE ENGINE CLASS
# ============================================================================

class IntelligenceEngine:
    """Main intelligence analysis engine"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.api = APIClient()
        self.patterns = {}
        self.threats = {}
        self.risks = {}
        self.analysis_history = []
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def initialize(self) -> bool:
        """Initialize intelligence engine"""
        try:
            # Connect to database
            if not self.db.connect():
                logger.error("Failed to connect to database")
                return False
            
            # Initialize API
            self._initialize_api()
            
            logger.info("Intelligence engine initialized")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _initialize_api(self) -> None:
        """Initialize API endpoints"""
        self.api.register_endpoint(
            'get_patterns',
            'GET',
            'intelligence/patterns',
            'Get detected patterns'
        )
        self.api.register_endpoint(
            'get_threats',
            'GET',
            'intelligence/threats',
            'Get detected threats'
        )
        self.api.register_endpoint(
            'get_risks',
            'GET',
            'intelligence/risks',
            'Get risk assessments'
        )
    
    # ========================================================================
    # PATTERN ANALYSIS
    # ========================================================================
    
    def analyze_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in data"""
        try:
            # Store in database
            pattern_record = self.db.create('patterns', {
                'data': data,
                'analysis_type': 'pattern_analysis',
                'status': 'analyzing'
            })
            
            # Analyze patterns
            patterns_found = self._detect_patterns(data)
            
            # Update record
            self.db.update('patterns', pattern_record['id'], {
                'patterns_found': patterns_found,
                'status': 'completed'
            })
            
            # Store in memory
            self.patterns[pattern_record['id']] = patterns_found
            
            # Log analysis
            analysis_record = {
                'type': 'pattern_analysis',
                'patterns_found': len(patterns_found),
                'timestamp': datetime.now().isoformat()
            }
            self.analysis_history.append(analysis_record)
            
            logger.info(f"Pattern analysis completed: {len(patterns_found)} patterns found")
            
            return {
                'success': True,
                'analysis_id': pattern_record['id'],
                'patterns': patterns_found,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _detect_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns in data"""
        patterns = []
        
        # Communication patterns
        if 'communications' in data:
            patterns.append({
                'type': 'communication_pattern',
                'description': 'Frequent communication with specific contacts',
                'confidence': 0.85
            })
        
        # Location patterns
        if 'locations' in data:
            patterns.append({
                'type': 'location_pattern',
                'description': 'Regular visits to specific locations',
                'confidence': 0.90
            })
        
        # Media patterns
        if 'media' in data:
            patterns.append({
                'type': 'media_pattern',
                'description': 'Specific media consumption patterns',
                'confidence': 0.75
            })
        
        return patterns
    
    # ========================================================================
    # THREAT DETECTION
    # ========================================================================
    
    def detect_threats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect threats in data"""
        try:
            # Store in database
            threat_record = self.db.create('threats', {
                'data': data,
                'analysis_type': 'threat_detection',
                'status': 'analyzing'
            })
            
            # Detect threats
            threats_found = self._identify_threats(data)
            
            # Update record
            self.db.update('threats', threat_record['id'], {
                'threats_found': threats_found,
                'status': 'completed'
            })
            
            # Store in memory
            self.threats[threat_record['id']] = threats_found
            
            # Log analysis
            analysis_record = {
                'type': 'threat_detection',
                'threats_found': len(threats_found),
                'timestamp': datetime.now().isoformat()
            }
            self.analysis_history.append(analysis_record)
            
            logger.info(f"Threat detection completed: {len(threats_found)} threats found")
            
            return {
                'success': True,
                'analysis_id': threat_record['id'],
                'threats': threats_found,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _identify_threats(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify threats in data"""
        threats = []
        
        # Check for suspicious communications
        if 'communications' in data:
            threats.append({
                'type': 'suspicious_communication',
                'severity': 'medium',
                'description': 'Suspicious communication patterns detected',
                'confidence': 0.75
            })
        
        # Check for suspicious locations
        if 'locations' in data:
            threats.append({
                'type': 'suspicious_location',
                'severity': 'low',
                'description': 'Unusual location visits detected',
                'confidence': 0.60
            })
        
        return threats
    
    # ========================================================================
    # RISK ASSESSMENT
    # ========================================================================
    
    def assess_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level"""
        try:
            # Store in database
            risk_record = self.db.create('risks', {
                'data': data,
                'analysis_type': 'risk_assessment',
                'status': 'analyzing'
            })
            
            # Assess risk
            risk_assessment = self._calculate_risk(data)
            
            # Update record
            self.db.update('risks', risk_record['id'], {
                'risk_assessment': risk_assessment,
                'status': 'completed'
            })
            
            # Store in memory
            self.risks[risk_record['id']] = risk_assessment
            
            # Log analysis
            analysis_record = {
                'type': 'risk_assessment',
                'risk_level': risk_assessment['level'],
                'timestamp': datetime.now().isoformat()
            }
            self.analysis_history.append(analysis_record)
            
            logger.info(f"Risk assessment completed: {risk_assessment['level']}")
            
            return {
                'success': True,
                'analysis_id': risk_record['id'],
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk level"""
        risk_score = 0
        
        # Calculate risk factors
        if 'communications' in data:
            risk_score += 20
        if 'locations' in data:
            risk_score += 15
        if 'media' in data:
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 40:
            level = 'high'
        elif risk_score >= 20:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': risk_score,
            'factors': ['communication_risk', 'location_risk', 'media_risk']
        }
    
    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================
    
    def run_comprehensive_analysis(self, case_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive intelligence analysis"""
        try:
            logger.info(f"Starting comprehensive analysis for case {case_id}")
            
            # Run all analyses
            patterns = self.analyze_patterns(data)
            threats = self.detect_threats(data)
            risks = self.assess_risk(data)
            
            # Compile results
            comprehensive_result = {
                'case_id': case_id,
                'patterns': patterns,
                'threats': threats,
                'risks': risks,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            # Store in database
            analysis_record = self.db.create('comprehensive_analysis', comprehensive_result)
            
            logger.info(f"Comprehensive analysis completed for case {case_id}")
            
            return {
                'success': True,
                'analysis_id': analysis_record['id'],
                'results': comprehensive_result
            }
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self.analysis_history[-limit:]
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_statistics()
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API statistics"""
        return self.api.get_statistics()
    
    def export_analysis_results(self, analysis_id: int) -> Dict[str, Any]:
        """Export analysis results"""
        try:
            # Get from database
            results = self.db.read('comprehensive_analysis', analysis_id)
            
            if results:
                return {
                    'success': True,
                    'data': results[0],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': 'Analysis not found'}
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'success': False, 'error': str(e)}

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_intelligence_engine() -> IntelligenceEngine:
    """Factory function to create intelligence engine"""
    return IntelligenceEngine()
