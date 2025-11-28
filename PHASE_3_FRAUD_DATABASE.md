# 🗄️ FRAUD & HARASSMENT DATABASE - PHASE 3 ENHANCEMENT

**Status**: New Feature - Fraud/Scammer Database
**Date**: November 25, 2025

---

## 🎯 CONCEPT

Add a **Fraud & Harassment Database** to both Suspicious Classifier and Location Intelligence modules to:
- ✅ Track known fraudsters/scammers
- ✅ Track known harassers
- ✅ Check phone numbers against database
- ✅ Check email addresses against database
- ✅ Check locations against database
- ✅ Alert when match found
- ✅ Crowdsource reports

---

## 📱 DATABASE STRUCTURE

### Table 1: FRAUDSTER NUMBERS

```python
fraudster_numbers = {
    "+1-555-0100": {
        "type": "PHISHING",
        "name": "Unknown Scammer",
        "reports": 45,
        "last_reported": "2025-11-24",
        "methods": ["Bank verification", "Account update"],
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    },
    "+1-555-0101": {
        "type": "ADVANCE_FEE",
        "name": "Romance Scammer",
        "reports": 23,
        "last_reported": "2025-11-25",
        "methods": ["Money request", "Emergency funds"],
        "risk_level": "HIGH",
        "status": "ACTIVE"
    },
    "+1-555-0102": {
        "type": "LOTTERY_SCAM",
        "name": "Lottery Scammer",
        "reports": 156,
        "last_reported": "2025-11-23",
        "methods": ["Prize claim", "Tax payment"],
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    }
}
```

---

### Table 2: FRAUDSTER EMAILS

```python
fraudster_emails = {
    "attacker@fake-bank.com": {
        "type": "PHISHING",
        "spoofs": "Bank of America",
        "reports": 89,
        "last_reported": "2025-11-25",
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    },
    "nigerian.prince@scam.com": {
        "type": "ADVANCE_FEE",
        "name": "Nigerian Prince Scam",
        "reports": 234,
        "last_reported": "2025-11-24",
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    },
    "support@paypa1.com": {
        "type": "PHISHING",
        "spoofs": "PayPal",
        "reports": 156,
        "last_reported": "2025-11-25",
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    }
}
```

---

### Table 3: HARASSER NUMBERS

```python
harasser_numbers = {
    "+1-555-0200": {
        "type": "HARASSMENT",
        "name": "John Doe (Harasser)",
        "reports": 12,
        "last_reported": "2025-11-25",
        "harassment_type": "CYBERBULLYING",
        "victims": 5,
        "risk_level": "HIGH",
        "status": "ACTIVE"
    },
    "+1-555-0201": {
        "type": "STALKING",
        "name": "Unknown Stalker",
        "reports": 8,
        "last_reported": "2025-11-24",
        "harassment_type": "STALKING",
        "victims": 3,
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    }
}
```

---

### Table 4: SUSPICIOUS LOCATIONS

```python
suspicious_locations = {
    "40.7128,-74.0060": {  # New York coordinates
        "location_name": "Scam Call Center",
        "type": "FRAUD_OPERATION",
        "reports": 45,
        "last_reported": "2025-11-25",
        "known_fraudsters": ["+1-555-0100", "+1-555-0101"],
        "risk_level": "CRITICAL",
        "status": "ACTIVE"
    },
    "40.7489,-73.9680": {  # Manhattan
        "location_name": "Harassment Hotspot",
        "type": "HARASSMENT_LOCATION",
        "reports": 23,
        "last_reported": "2025-11-24",
        "known_harassers": ["+1-555-0200"],
        "risk_level": "HIGH",
        "status": "ACTIVE"
    }
}
```

---

### Table 5: FRAUD PATTERNS

```python
fraud_patterns = {
    "BANK_PHISHING": {
        "keywords": ["verify", "confirm", "update", "account", "urgent"],
        "common_senders": ["attacker@fake-bank.com"],
        "common_numbers": ["+1-555-0100"],
        "risk_score": 0.95,
        "reports": 234
    },
    "ROMANCE_SCAM": {
        "keywords": ["love", "money", "emergency", "help", "wire"],
        "common_senders": ["romance.scammer@email.com"],
        "common_numbers": ["+1-555-0101"],
        "risk_score": 0.88,
        "reports": 156
    },
    "LOTTERY_SCAM": {
        "keywords": ["won", "prize", "claim", "tax", "payment"],
        "common_senders": ["lottery@scam.com"],
        "common_numbers": ["+1-555-0102"],
        "risk_score": 0.92,
        "reports": 189
    }
}
```

---

## 🔍 FEATURES FOR SUSPICIOUS CLASSIFIER

### Feature 1: PHONE NUMBER CHECK ✅

**What it does:**
- Checks incoming phone number against database
- Returns match if found
- Shows fraud history

**Implementation:**
```python
def check_phone_number(phone_number):
    if phone_number in fraudster_numbers:
        return {
            "match": True,
            "type": fraudster_numbers[phone_number]["type"],
            "reports": fraudster_numbers[phone_number]["reports"],
            "risk_level": fraudster_numbers[phone_number]["risk_level"],
            "methods": fraudster_numbers[phone_number]["methods"]
        }
    elif phone_number in harasser_numbers:
        return {
            "match": True,
            "type": "HARASSMENT",
            "reports": harasser_numbers[phone_number]["reports"],
            "risk_level": harasser_numbers[phone_number]["risk_level"]
        }
    return {"match": False}
```

**Output:**
```json
{
  "phone_number": "+1-555-0100",
  "database_match": true,
  "type": "PHISHING",
  "reports": 45,
  "risk_level": "CRITICAL",
  "methods": ["Bank verification", "Account update"],
  "recommendation": "BLOCK - Known fraudster"
}
```

---

### Feature 2: EMAIL CHECK ✅

**What it does:**
- Checks sender email against database
- Returns match if found
- Shows spoofed organization

**Implementation:**
```python
def check_email(email):
    if email in fraudster_emails:
        return {
            "match": True,
            "type": fraudster_emails[email]["type"],
            "spoofs": fraudster_emails[email].get("spoofs", "Unknown"),
            "reports": fraudster_emails[email]["reports"],
            "risk_level": fraudster_emails[email]["risk_level"]
        }
    return {"match": False}
```

**Output:**
```json
{
  "email": "attacker@fake-bank.com",
  "database_match": true,
  "type": "PHISHING",
  "spoofs": "Bank of America",
  "reports": 89,
  "risk_level": "CRITICAL",
  "recommendation": "BLOCK - Known phishing email"
}
```

---

### Feature 3: PATTERN MATCHING ✅

**What it does:**
- Matches message against known fraud patterns
- Calculates pattern similarity
- Returns matching patterns

**Implementation:**
```python
def match_fraud_patterns(message):
    matches = []
    for pattern_name, pattern_data in fraud_patterns.items():
        keyword_matches = sum(1 for kw in pattern_data["keywords"] if kw.lower() in message.lower())
        if keyword_matches > 0:
            similarity = keyword_matches / len(pattern_data["keywords"])
            matches.append({
                "pattern": pattern_name,
                "similarity": similarity,
                "risk_score": pattern_data["risk_score"],
                "reports": pattern_data["reports"]
            })
    return sorted(matches, key=lambda x: x["similarity"], reverse=True)
```

**Output:**
```json
{
  "message": "Click to verify your bank account",
  "pattern_matches": [
    {
      "pattern": "BANK_PHISHING",
      "similarity": 0.80,
      "risk_score": 0.95,
      "reports": 234
    }
  ]
}
```

---

### Feature 4: COMBINED RISK SCORE ✅

**What it does:**
- Combines database checks with AI analysis
- Calculates final risk score
- Provides recommendation

**Implementation:**
```python
def calculate_combined_risk(message, phone, email):
    ai_risk = get_ai_classification_score(message)
    
    phone_risk = 0.95 if check_phone_number(phone)["match"] else 0
    email_risk = 0.95 if check_email(email)["match"] else 0
    pattern_risk = get_pattern_match_score(message)
    
    combined_risk = (ai_risk * 0.4 + phone_risk * 0.2 + 
                     email_risk * 0.2 + pattern_risk * 0.2)
    
    return {
        "ai_risk": ai_risk,
        "phone_risk": phone_risk,
        "email_risk": email_risk,
        "pattern_risk": pattern_risk,
        "combined_risk": combined_risk,
        "recommendation": get_recommendation(combined_risk)
    }
```

**Output:**
```json
{
  "ai_risk": 0.85,
  "phone_risk": 0.95,
  "email_risk": 0.95,
  "pattern_risk": 0.80,
  "combined_risk": 0.89,
  "recommendation": "CRITICAL - Block immediately"
}
```

---

## 📍 FEATURES FOR LOCATION INTELLIGENCE

### Feature 1: LOCATION RISK CHECK ✅

**What it does:**
- Checks GPS coordinates against suspicious locations
- Returns risk level
- Shows known fraudsters at location

**Implementation:**
```python
def check_location_risk(latitude, longitude):
    location_key = f"{latitude},{longitude}"
    
    if location_key in suspicious_locations:
        return {
            "match": True,
            "location_name": suspicious_locations[location_key]["location_name"],
            "type": suspicious_locations[location_key]["type"],
            "reports": suspicious_locations[location_key]["reports"],
            "risk_level": suspicious_locations[location_key]["risk_level"],
            "known_fraudsters": suspicious_locations[location_key]["known_fraudsters"]
        }
    return {"match": False}
```

**Output:**
```json
{
  "coordinates": [40.7128, -74.0060],
  "location_match": true,
  "location_name": "Scam Call Center",
  "type": "FRAUD_OPERATION",
  "reports": 45,
  "risk_level": "CRITICAL",
  "known_fraudsters": ["+1-555-0100", "+1-555-0101"],
  "recommendation": "ALERT - Known fraud location"
}
```

---

### Feature 2: FRAUDSTER LOCATION TRACKING ✅

**What it does:**
- Tracks known fraudsters' movements
- Alerts if fraudster detected at location
- Shows movement patterns

**Implementation:**
```python
def track_fraudster_location(phone_number, latitude, longitude):
    if phone_number in fraudster_numbers:
        fraudster = fraudster_numbers[phone_number]
        
        return {
            "fraudster_detected": True,
            "fraudster_name": fraudster["name"],
            "type": fraudster["type"],
            "location": [latitude, longitude],
            "timestamp": datetime.now(),
            "risk_level": fraudster["risk_level"],
            "recommendation": "ALERT - Fraudster location detected"
        }
    return {"fraudster_detected": False}
```

**Output:**
```json
{
  "fraudster_detected": true,
  "fraudster_name": "Unknown Scammer",
  "phone_number": "+1-555-0100",
  "location": [40.7128, -74.0060],
  "timestamp": "2025-11-25 14:30:00",
  "risk_level": "CRITICAL",
  "recommendation": "ALERT - Known fraudster at location"
}
```

---

### Feature 3: HARASSER LOCATION TRACKING ✅

**What it does:**
- Tracks known harassers' movements
- Alerts if harasser near victim
- Shows proximity warnings

**Implementation:**
```python
def check_harasser_proximity(harasser_phone, victim_location, radius_miles=5):
    if harasser_phone in harasser_numbers:
        harasser = harasser_numbers[harasser_phone]
        
        # Calculate distance (simplified)
        distance = calculate_distance(harasser_location, victim_location)
        
        if distance < radius_miles:
            return {
                "harasser_nearby": True,
                "harasser_name": harasser["name"],
                "distance": distance,
                "risk_level": "CRITICAL",
                "recommendation": "ALERT - Harasser nearby"
            }
    return {"harasser_nearby": False}
```

**Output:**
```json
{
  "harasser_nearby": true,
  "harasser_name": "John Doe (Harasser)",
  "distance": 2.5,
  "distance_unit": "miles",
  "risk_level": "CRITICAL",
  "recommendation": "ALERT - Harasser within 5 miles"
}
```

---

### Feature 4: LOCATION ANOMALY WITH DATABASE ✅

**What it does:**
- Combines location anomalies with database
- Flags unusual visits to fraud locations
- Tracks suspicious movement patterns

**Implementation:**
```python
def analyze_location_with_database(locations_history):
    alerts = []
    
    for location in locations_history:
        # Check if location is suspicious
        location_check = check_location_risk(location["lat"], location["lon"])
        
        if location_check["match"]:
            alerts.append({
                "type": "SUSPICIOUS_LOCATION_VISIT",
                "location": location_check["location_name"],
                "timestamp": location["timestamp"],
                "risk_level": location_check["risk_level"]
            })
    
    return alerts
```

**Output:**
```json
{
  "location_alerts": [
    {
      "type": "SUSPICIOUS_LOCATION_VISIT",
      "location": "Scam Call Center",
      "timestamp": "2025-11-25 14:30:00",
      "risk_level": "CRITICAL"
    }
  ]
}
```

---

## 🗄️ DATABASE MANAGEMENT

### Feature 1: ADD NEW FRAUDSTER ✅

**UI:**
```
[Add Fraudster]
Phone: +1-555-0103
Type: PHISHING
Name: New Scammer
Methods: [Bank verification] [Account update]
Risk Level: [CRITICAL] [HIGH] [MEDIUM]
[Submit]
```

**Backend:**
```python
def add_fraudster(phone, fraud_type, name, methods, risk_level):
    fraudster_numbers[phone] = {
        "type": fraud_type,
        "name": name,
        "reports": 1,
        "last_reported": datetime.now(),
        "methods": methods,
        "risk_level": risk_level,
        "status": "ACTIVE"
    }
    return {"status": "Added", "phone": phone}
```

---

### Feature 2: REPORT FRAUDSTER ✅

**What it does:**
- Users can report new fraudsters
- Increments report count
- Updates last reported date

**Implementation:**
```python
def report_fraudster(phone_number, fraud_type, description):
    if phone_number in fraudster_numbers:
        fraudster_numbers[phone_number]["reports"] += 1
        fraudster_numbers[phone_number]["last_reported"] = datetime.now()
    else:
        add_fraudster(phone_number, fraud_type, "Unknown", [], "HIGH")
    
    return {"status": "Reported", "reports": fraudster_numbers[phone_number]["reports"]}
```

---

### Feature 3: STATISTICS & ANALYTICS ✅

**What it does:**
- Shows database statistics
- Displays top fraudsters
- Shows fraud trends

**Output:**
```json
{
  "total_fraudsters": 1245,
  "total_harassers": 345,
  "total_reports": 5678,
  "top_fraudsters": [
    {
      "phone": "+1-555-0102",
      "type": "LOTTERY_SCAM",
      "reports": 156
    },
    {
      "phone": "+1-555-0101",
      "type": "ADVANCE_FEE",
      "reports": 89
    }
  ],
  "fraud_types": {
    "PHISHING": 234,
    "ADVANCE_FEE": 156,
    "LOTTERY_SCAM": 189
  }
}
```

---

## 🎯 UI INTEGRATION

### Suspicious Classifier UI:

```
┌─────────────────────────────────────────────┐
│ 🔍 SUSPICIOUS CLASSIFIER                    │
├─────────────────────────────────────────────┤
│                                             │
│ Message: "Click to verify account"         │
│ From: +1-555-0100                          │
│ Email: attacker@fake-bank.com              │
│                                             │
│ Analysis Results:                           │
│ ┌─────────────────────────────────────────┐ │
│ │ AI Classification: PHISHING (95%)       │ │
│ │ Database Match: ✅ YES                  │ │
│ │   - Phone: Known Fraudster (45 reports)│ │
│ │   - Email: Known Phishing (89 reports) │ │
│ │   - Pattern: Bank Phishing Match        │ │
│ │                                         │ │
│ │ Combined Risk: CRITICAL (89%)           │ │
│ │ Recommendation: BLOCK IMMEDIATELY       │ │
│ │                                         │ │
│ │ [View Details] [Report] [Block]        │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Location Intelligence UI:

```
┌─────────────────────────────────────────────┐
│ 📍 LOCATION INTELLIGENCE                    │
├─────────────────────────────────────────────┤
│                                             │
│ Location: 40.7128, -74.0060 (New York)    │
│                                             │
│ Location Analysis:                          │
│ ┌─────────────────────────────────────────┐ │
│ │ Database Match: ✅ YES                  │ │
│ │   - Location: Scam Call Center          │ │
│ │   - Type: FRAUD_OPERATION               │ │
│ │   - Reports: 45                         │ │
│ │   - Risk Level: CRITICAL                │ │
│ │                                         │ │
│ │ Known Fraudsters at Location:           │ │
│ │   - +1-555-0100 (Phishing)             │ │
│ │   - +1-555-0101 (Advance Fee)          │ │
│ │                                         │ │
│ │ Recommendation: ALERT - Fraud Location │ │
│ │                                         │ │
│ │ [View Details] [Report] [Alert]        │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 📊 DATABASE SCHEMA

### Tables to Create:

```sql
-- Fraudster Phone Numbers
CREATE TABLE fraudster_phones (
    id INT PRIMARY KEY,
    phone_number VARCHAR(20),
    fraud_type VARCHAR(50),
    name VARCHAR(100),
    reports INT,
    last_reported DATETIME,
    methods JSON,
    risk_level VARCHAR(20),
    status VARCHAR(20)
);

-- Fraudster Emails
CREATE TABLE fraudster_emails (
    id INT PRIMARY KEY,
    email VARCHAR(100),
    fraud_type VARCHAR(50),
    spoofs VARCHAR(100),
    reports INT,
    last_reported DATETIME,
    risk_level VARCHAR(20),
    status VARCHAR(20)
);

-- Harasser Numbers
CREATE TABLE harasser_numbers (
    id INT PRIMARY KEY,
    phone_number VARCHAR(20),
    name VARCHAR(100),
    reports INT,
    last_reported DATETIME,
    harassment_type VARCHAR(50),
    victims INT,
    risk_level VARCHAR(20),
    status VARCHAR(20)
);

-- Suspicious Locations
CREATE TABLE suspicious_locations (
    id INT PRIMARY KEY,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    location_name VARCHAR(100),
    location_type VARCHAR(50),
    reports INT,
    last_reported DATETIME,
    known_fraudsters JSON,
    risk_level VARCHAR(20),
    status VARCHAR(20)
);

-- Fraud Patterns
CREATE TABLE fraud_patterns (
    id INT PRIMARY KEY,
    pattern_name VARCHAR(100),
    keywords JSON,
    common_senders JSON,
    common_numbers JSON,
    risk_score DECIMAL(3, 2),
    reports INT
);
```

---

## ✅ BENEFITS

✅ **Real-time Detection** - Instant match against database
✅ **Crowdsourced Data** - Community reports
✅ **Pattern Recognition** - AI + Database combined
✅ **Location Tracking** - Know fraud hotspots
✅ **Harasser Alerts** - Proximity warnings
✅ **Comprehensive Risk** - Multiple data sources
✅ **Actionable Intelligence** - Clear recommendations

---

## 🚀 IMPLEMENTATION PLAN

### Phase 3a: Database Setup
- [ ] Create database tables
- [ ] Populate initial fraudster data
- [ ] Populate harasser data
- [ ] Populate suspicious locations

### Phase 3b: Suspicious Classifier Integration
- [ ] Add phone number check
- [ ] Add email check
- [ ] Add pattern matching
- [ ] Add combined risk scoring

### Phase 3c: Location Intelligence Integration
- [ ] Add location risk check
- [ ] Add fraudster tracking
- [ ] Add harasser proximity
- [ ] Add location anomaly detection

### Phase 3d: Database Management
- [ ] Add fraudster reporting UI
- [ ] Add statistics dashboard
- [ ] Add data management tools
- [ ] Add crowdsourcing features

---

## ✅ FRAUD & HARASSMENT DATABASE READY

**Status**: Feature designed and ready to implement
**Impact**: Significantly enhances detection accuracy
**Timeline**: 3-4 days additional development
