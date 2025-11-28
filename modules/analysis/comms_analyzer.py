"""
COMMS ANALYZER - Communications Analyzer
Analyzes messages, calls, and communications for suspicious patterns
Integrated with fraud database for real-time detection

This module provides:
- Message classification (phishing, fraud, threats, spam)
- Keyword detection
- Entity extraction
- Phishing detection
- Threat detection
- Fraud detection
- Sentiment analysis
- Pattern analysis
- Risk scoring
- Database integration (fraudster/harasser lookup)
- Auto-reporting to database
"""

import logging
import re
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from transformers import pipeline
import requests

from modules.analysis.models import DatabaseManager, updater
from modules.shared.utils import ArtifactPathBuilder, ResultsRepository

logger = logging.getLogger(__name__)

# ============================================================================
# SUSPICIOUS KEYWORDS DATABASE
# ============================================================================

SUSPICIOUS_KEYWORDS = {
    "PHISHING": [
        "verify", "confirm", "update", "account", "urgent", "click here",
        "click link", "validate", "authenticate", "re-enter", "suspended",
        "limited time", "act now", "confirm identity", "verify password"
    ],
    "FRAUD": [
        "wire", "transfer", "payment", "money", "send", "bank account",
        "credit card", "social security", "tax id", "routing number",
        "swift code", "account number", "inheritance", "prize"
    ],
    "THREAT": [
        "kill", "hurt", "attack", "bomb", "weapon", "shoot", "stab",
        "rape", "assault", "violence", "harm", "danger", "threat",
        "die", "dead", "murder", "destroy"
    ],
    "SPAM": [
        "free", "limited time", "act now", "click here", "buy now",
        "special offer", "exclusive deal", "winner", "congratulations",
        "claim prize", "unsubscribe"
    ]
}

# ============================================================================
# FRAUD PATTERNS
# ============================================================================

FRAUD_PATTERNS = {
    "BANK_PHISHING": {
        "keywords": ["verify", "confirm", "update", "account", "urgent"],
        "risk_score": 0.95,
        "description": "Bank account verification phishing"
    },
    "PAYPAL_PHISHING": {
        "keywords": ["paypal", "verify", "account", "confirm", "update"],
        "risk_score": 0.92,
        "description": "PayPal phishing attempt"
    },
    "AMAZON_PHISHING": {
        "keywords": ["amazon", "account", "suspended", "verify", "confirm"],
        "risk_score": 0.90,
        "description": "Amazon account phishing"
    },
    "ROMANCE_SCAM": {
        "keywords": ["love", "money", "emergency", "help", "wire", "transfer"],
        "risk_score": 0.88,
        "description": "Romance/dating scam"
    },
    "LOTTERY_SCAM": {
        "keywords": ["won", "prize", "claim", "tax", "payment", "lottery"],
        "risk_score": 0.92,
        "description": "Lottery/prize scam"
    },
    "TECH_SUPPORT_SCAM": {
        "keywords": ["virus", "malware", "error", "support", "call", "remote"],
        "risk_score": 0.85,
        "description": "Tech support scam"
    },
    "IRS_SCAM": {
        "keywords": ["irs", "tax", "debt", "legal", "arrest", "warrant"],
        "risk_score": 0.93,
        "description": "IRS/tax authority scam"
    }
}

# ============================================================================
# SUSPICIOUS CLASSIFIER CLASS
# ============================================================================

class CommsAnalyzer:
    """Communications Analyzer - Analyze messages for suspicious content with database integration"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize analyzer with transformers and database connection"""
        self.api_url = api_url
        self.db = DatabaseManager()
        self.updater = updater
        
        # Initialize transformers
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logger.info("✅ Zero-shot classifier loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load zero-shot classifier: {e}")
            self.zero_shot_classifier = None
        
        try:
            self.ner = pipeline(
                "ner",
                model="dslim/bert-base-multilingual-cased-ner"
            )
            logger.info("✅ NER model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load NER model: {e}")
            self.ner = None
        
        try:
            self.sentiment = pipeline("sentiment-analysis")
            logger.info("✅ Sentiment analyzer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment analyzer: {e}")
            self.sentiment = None
    
    # ========================================================================
    # KEYWORD DETECTION
    # ========================================================================
    
    def detect_keywords(self, message: str) -> Dict[str, Any]:
        """Detect suspicious keywords in message"""
        message_lower = message.lower()
        found_keywords = {}
        
        for category, keywords in SUSPICIOUS_KEYWORDS.items():
            matches = []
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    matches.append(keyword)
            
            if matches:
                found_keywords[category] = {
                    "keywords": matches,
                    "count": len(matches),
                    "risk_increase": len(matches) * 0.1
                }
        
        return {
            "found": len(found_keywords) > 0,
            "categories": found_keywords,
            "total_matches": sum(cat["count"] for cat in found_keywords.values())
        }
    
    # ========================================================================
    # PATTERN MATCHING
    # ========================================================================
    
    def match_fraud_patterns(self, message: str) -> List[Dict[str, Any]]:
        """Match message against known fraud patterns"""
        message_lower = message.lower()
        matches = []
        
        for pattern_name, pattern_data in FRAUD_PATTERNS.items():
            keyword_matches = sum(
                1 for kw in pattern_data["keywords"] 
                if kw.lower() in message_lower
            )
            
            if keyword_matches > 0:
                similarity = keyword_matches / len(pattern_data["keywords"])
                matches.append({
                    "pattern": pattern_name,
                    "similarity": round(similarity, 2),
                    "risk_score": pattern_data["risk_score"],
                    "description": pattern_data["description"],
                    "keyword_matches": keyword_matches
                })
        
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)
    
    # ========================================================================
    # ENTITY EXTRACTION
    # ========================================================================
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message using NER"""
        if not self.ner:
            return {"error": "NER model not available"}
        
        try:
            entities = self.ner(message)
            
            # Group entities by type
            grouped = {}
            for entity in entities:
                entity_type = entity["entity"].replace("B-", "").replace("I-", "")
                if entity_type not in grouped:
                    grouped[entity_type] = []
                grouped[entity_type].append(entity["word"])
            
            return {
                "found": len(entities) > 0,
                "entities": grouped,
                "total_entities": len(entities)
            }
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # PHISHING DETECTION
    # ========================================================================
    
    def detect_phishing(self, message: str, sender: str = None) -> Dict[str, Any]:
        """Detect phishing attempts"""
        phishing_indicators = []
        phishing_score = 0.0
        
        # Check keywords
        keywords = self.detect_keywords(message)
        if "PHISHING" in keywords.get("categories", {}):
            phishing_indicators.append("Phishing keywords detected")
            phishing_score += 0.3
        
        # Check patterns
        patterns = self.match_fraud_patterns(message)
        phishing_patterns = [p for p in patterns if "PHISHING" in p["pattern"]]
        if phishing_patterns:
            phishing_indicators.append(f"Matches {phishing_patterns[0]['pattern']}")
            phishing_score += phishing_patterns[0]["risk_score"] * 0.4
        
        # Check urgency
        urgency_keywords = ["urgent", "immediate", "act now", "limited time"]
        if any(kw in message.lower() for kw in urgency_keywords):
            phishing_indicators.append("Urgency language detected")
            phishing_score += 0.2
        
        # Check sender spoofing
        if sender:
            if "@" in sender:
                domain = sender.split("@")[1]
                suspicious_domains = ["fake-", "verify-", "confirm-", "update-"]
                if any(d in domain for d in suspicious_domains):
                    phishing_indicators.append("Suspicious sender domain")
                    phishing_score += 0.25
        
        return {
            "phishing_detected": phishing_score > 0.5,
            "phishing_score": round(min(phishing_score, 1.0), 2),
            "indicators": phishing_indicators,
            "recommendation": "BLOCK" if phishing_score > 0.7 else "REVIEW" if phishing_score > 0.4 else "SAFE"
        }
    
    # ========================================================================
    # THREAT DETECTION
    # ========================================================================
    
    def detect_threats(self, message: str) -> Dict[str, Any]:
        """Detect threats and violence"""
        threat_indicators = []
        threat_score = 0.0
        
        # Check threat keywords
        keywords = self.detect_keywords(message)
        if "THREAT" in keywords.get("categories", {}):
            threat_indicators.append("Threat keywords detected")
            threat_score += 0.5
        
        # Check sentiment
        if self.sentiment:
            try:
                sentiment = self.sentiment(message)[0]
                if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.9:
                    threat_indicators.append("Highly negative sentiment")
                    threat_score += 0.3
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Check intensity
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        if caps_ratio > 0.5:
            threat_indicators.append("Excessive capitalization")
            threat_score += 0.2
        
        # Check exclamation marks
        exclamation_count = message.count("!")
        if exclamation_count > 3:
            threat_indicators.append("Multiple exclamation marks")
            threat_score += 0.15
        
        return {
            "threat_detected": threat_score > 0.5,
            "threat_score": round(min(threat_score, 1.0), 2),
            "severity": "CRITICAL" if threat_score > 0.8 else "HIGH" if threat_score > 0.6 else "MEDIUM" if threat_score > 0.4 else "LOW",
            "indicators": threat_indicators
        }
    
    # ========================================================================
    # FRAUD DETECTION
    # ========================================================================
    
    def detect_fraud(self, message: str) -> Dict[str, Any]:
        """Detect financial fraud attempts"""
        fraud_indicators = []
        fraud_score = 0.0
        
        # Check keywords
        keywords = self.detect_keywords(message)
        if "FRAUD" in keywords.get("categories", {}):
            fraud_indicators.append("Fraud keywords detected")
            fraud_score += 0.4
        
        # Check patterns
        patterns = self.match_fraud_patterns(message)
        fraud_patterns = [p for p in patterns if any(
            x in p["pattern"] for x in ["ROMANCE", "LOTTERY", "ADVANCE"]
        )]
        if fraud_patterns:
            fraud_indicators.append(f"Matches {fraud_patterns[0]['pattern']}")
            fraud_score += fraud_patterns[0]["risk_score"] * 0.3
        
        # Check money requests
        money_keywords = ["send", "wire", "transfer", "payment", "money"]
        if any(kw in message.lower() for kw in money_keywords):
            fraud_indicators.append("Money request detected")
            fraud_score += 0.3
        
        return {
            "fraud_detected": fraud_score > 0.5,
            "fraud_score": round(min(fraud_score, 1.0), 2),
            "indicators": fraud_indicators,
            "recommendation": "ALERT" if fraud_score > 0.7 else "REVIEW" if fraud_score > 0.4 else "SAFE"
        }
    
    # ========================================================================
    # DATABASE INTEGRATION
    # ========================================================================
    
    def check_phone_database(self, phone: str) -> Dict[str, Any]:
        """Check phone against fraudster/harasser database"""
        try:
            # Check fraudster database
            fraudster = self.db.get_fraudster(phone)
            if fraudster:
                logger.info(f"✅ Found fraudster in database: {phone}")
                return {
                    "match": True,
                    "type": "FRAUDSTER",
                    "fraud_type": fraudster.fraud_type,
                    "reports": fraudster.reports,
                    "risk_level": fraudster.risk_level,
                    "status": fraudster.status,
                    "name": fraudster.name
                }
            
            # Check harasser database
            harasser = self.db.get_harasser(phone)
            if harasser:
                logger.info(f"✅ Found harasser in database: {phone}")
                return {
                    "match": True,
                    "type": "HARASSER",
                    "harassment_type": harasser.harassment_type,
                    "reports": harasser.reports,
                    "risk_level": harasser.risk_level,
                    "status": harasser.status,
                    "name": harasser.name
                }
        except Exception as e:
            logger.warning(f"Database check failed: {e}")
        
        return {"match": False}
    
    def check_email_database(self, email: str) -> Dict[str, Any]:
        """Check email against fraudster email database"""
        try:
            # Query fraudster emails from database
            # This would need a method in DatabaseManager
            # For now, return not found
            return {"match": False}
        except Exception as e:
            logger.warning(f"Email database check failed: {e}")
            return {"match": False}
    
    # ========================================================================
    # COMBINED ANALYSIS
    # ========================================================================
    
    def analyze_message(self, message: str, phone: str = None, 
                       contact_phone: str = None, email: str = None, sender_name: str = None,
                       case_id: str = None, consent_manager: Any = None) -> Dict[str, Any]:
        """Comprehensive message analysis with contact phone tracking and consent verification"""
        
        # Check consent if available
        if consent_manager and case_id:
            try:
                from modules.consent.models import ConsentLevel, MODULE_MIN_LEVELS
                
                session = consent_manager.get_session(case_id)
                if session:
                    min_level = MODULE_MIN_LEVELS.get('communications', ConsentLevel.LEGAL)
                    
                    if session.level < min_level:
                        logger.warning(f"Communications analysis blocked: {session.level.name} < {min_level.name}")
                        return {
                            'status': 'consent_denied',
                            'message': f'Communications analysis requires {min_level.name} consent',
                            'required_level': min_level.name,
                            'current_level': session.level.name,
                            'case_id': case_id
                        }
            except Exception as e:
                logger.error(f"Error checking consent: {e}")
        
        # Run all analyses
        keywords = self.detect_keywords(message)
        patterns = self.match_fraud_patterns(message)
        entities = self.extract_entities(message)
        phishing = self.detect_phishing(message, email)
        threats = self.detect_threats(message)
        fraud = self.detect_fraud(message)
        
        # Database checks
        phone_check = self.check_phone_database(phone) if phone else {"match": False}
        email_check = self.check_email_database(email) if email else {"match": False}
        contact_phone_check = self.check_phone_database(contact_phone) if contact_phone else {"match": False}
        
        # Auto-report if found in database
        if phone_check["match"] and phone:
            if phone_check["type"] == "FRAUDSTER":
                self.updater.auto_report_fraudster(phone)
                logger.info(f"✅ Auto-reported fraudster: {phone}")
            elif phone_check["type"] == "HARASSER":
                self.updater.auto_report_harasser(phone)
                logger.info(f"✅ Auto-reported harasser: {phone}")
        
        # Calculate combined risk score
        ai_risk = max(
            phishing["phishing_score"],
            threats["threat_score"],
            fraud["fraud_score"]
        )
        
        db_risk = 0.95 if (phone_check["match"] or email_check["match"]) else 0.0
        
        combined_risk = (ai_risk * 0.6 + db_risk * 0.4)
        
        # Determine classification
        if combined_risk > 0.85:
            classification = "CRITICAL"
        elif combined_risk > 0.70:
            classification = "HIGH"
        elif combined_risk > 0.50:
            classification = "MEDIUM"
        else:
            classification = "LOW"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "message_preview": message[:100] + "..." if len(message) > 100 else message,
            "sender": {
                "phone": phone,
                "email": email,
                "name": sender_name
            },
            "contact": {
                "phone": contact_phone,
                "phone_match": contact_phone_check
            },
            "analysis": {
                "keywords": keywords,
                "patterns": patterns[:3],  # Top 3 patterns
                "entities": entities,
                "phishing": phishing,
                "threats": threats,
                "fraud": fraud
            },
            "database_checks": {
                "sender_phone_match": phone_check,
                "contact_phone_match": contact_phone_check,
                "email_match": email_check
            },
            "risk_scores": {
                "ai_risk": round(ai_risk, 2),
                "database_risk": round(db_risk, 2),
                "combined_risk": round(combined_risk, 2)
            },
            "classification": classification,
            "recommendation": self._get_recommendation(classification, phishing, threats, fraud),
            "actions": self._get_actions(classification, phone_check, email_check)
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_recommendation(self, classification: str, phishing: Dict, 
                           threats: Dict, fraud: Dict) -> str:
        """Get recommendation based on analysis"""
        if classification == "CRITICAL":
            if threats["threat_detected"]:
                return "ALERT AUTHORITIES - Threat detected"
            elif phishing["phishing_detected"]:
                return "BLOCK - Known phishing attempt"
            else:
                return "BLOCK IMMEDIATELY - Critical risk"
        elif classification == "HIGH":
            if phishing["phishing_detected"]:
                return "BLOCK - Likely phishing"
            elif fraud["fraud_detected"]:
                return "ALERT - Likely fraud"
            else:
                return "REVIEW CAREFULLY"
        elif classification == "MEDIUM":
            return "REVIEW - Suspicious content"
        else:
            return "SAFE - No threats detected"
    
    def _get_actions(self, classification: str, phone_check: Dict, 
                    email_check: Dict) -> List[str]:
        """Get recommended actions"""
        actions = []
        
        if classification == "CRITICAL":
            actions.append("Block sender")
            actions.append("Report to authorities")
            actions.append("Alert user")
        elif classification == "HIGH":
            actions.append("Block or quarantine")
            actions.append("Flag for review")
            actions.append("Notify user")
        elif classification == "MEDIUM":
            actions.append("Flag for review")
            actions.append("Monitor for patterns")
        
        if phone_check["match"]:
            actions.append(f"Known fraudster: {phone_check.get('type')}")
        
        if email_check["match"]:
            actions.append(f"Known phishing email: {email_check.get('type')}")
        
        return actions
    
    # ========================================================================
    # ARTIFACT ROUTING
    # ========================================================================
    
    def save_analysis_results(self, case_id: str, results: Dict[str, Any]) -> bool:
        """Save analysis results to artifact storage"""
        try:
            # Resolve artifact path
            artifact_path = ArtifactPathBuilder.resolve(
                case_id, 
                "analysis", 
                ensure_dir=True
            )
            
            # Save comms analysis
            comms_file = os.path.join(artifact_path, "comms_analysis.json")
            
            with open(comms_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Comms analysis saved to {comms_file}")
            
            # Also save to results repository
            ResultsRepository.save(case_id, {"comms_analysis": results})
            
            return True
        except Exception as e:
            logger.error(f"❌ Error saving comms analysis: {e}")
            return False
    
    def load_analysis_results(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Load analysis results from artifact storage"""
        try:
            artifact_path = ArtifactPathBuilder.resolve(case_id, "analysis")
            comms_file = os.path.join(artifact_path, "comms_analysis.json")
            
            if os.path.exists(comms_file):
                with open(comms_file, 'r') as f:
                    results = json.load(f)
                
                logger.info(f"✅ Comms analysis loaded from {comms_file}")
                return results
            
            return None
        except Exception as e:
            logger.error(f"❌ Error loading comms analysis: {e}")
            return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    analyzer = CommsAnalyzer()
    
    # Test messages
    test_messages = [
        {
            "message": "Click here to verify your bank account immediately!",
            "phone": "+1-555-0100",
            "email": "verify@fake-bank.com"
        },
        {
            "message": "I'm going to hurt you",
            "phone": "+1-555-0200"
        },
        {
            "message": "You won a prize! Send $100 to claim.",
            "phone": "+1-555-0102"
        }
    ]
    
    for test in test_messages:
        print(f"\n{'='*60}")
        print(f"Analyzing: {test['message']}")
        print(f"{'='*60}")
        
        result = analyzer.analyze_message(
            message=test["message"],
            phone=test.get("phone"),
            email=test.get("email")
        )
        
        print(f"Classification: {result['classification']}")
        print(f"Combined Risk: {result['risk_scores']['combined_risk']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Actions: {result['actions']}")
        print(f"Database Match: {result['database_checks']['phone_match'].get('match', False)}")
