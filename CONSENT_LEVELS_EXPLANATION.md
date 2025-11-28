# 🔐 CONSENT LEVELS - COMPLETE EXPLANATION

**Status**: UPDATED TO 3 LEVELS ONLY
**Date**: November 25, 2025

---

## 📋 THREE CONSENT LEVELS

Forensmart uses **3 consent levels** to control data access during forensic investigations:

```
STANDARD (Level 1)  ← Basic forensics
    ↓
LEGAL (Level 2)     ← Legal investigation
    ↓
FULL (Level 3)      ← Comprehensive forensics
```

---

## 🟡 LEVEL 1: STANDARD

**Value**: 1
**Color**: Yellow 🟡
**Purpose**: Basic forensic analysis

### What's Included:
- ✅ Device Information (hardware, OS, model, IMEI, etc.)
- ✅ Location Data (GPS coordinates, cell tower data, location history)
- ✅ Media Files (photos, videos, audio files)
- ✅ Security Settings (password strength, authentication methods, security apps)

### What's NOT Included:
- ❌ Communications (SMS, calls, messaging apps)
- ❌ System Logs (system diagnostics, kernel logs)

### Use Cases:
- Basic device profiling
- Location pattern analysis
- Media content review
- Security posture assessment

### Example Scenario:
```
Investigator needs to:
- Track suspect's location history
- Review photos/videos
- Check device security settings
- Get device specifications

→ STANDARD consent is sufficient
```

---

## 🟠 LEVEL 2: LEGAL

**Value**: 2
**Color**: Orange 🟠
**Purpose**: Legal investigation with communications

### What's Included:
- ✅ Everything from STANDARD level
- ✅ Communications (SMS, call logs, contacts, messaging apps)
- ✅ Call Records (incoming, outgoing, missed calls)
- ✅ Message History (text messages, instant messages)
- ✅ Contact Information (phone numbers, email addresses)

### What's NOT Included:
- ❌ System Logs (system diagnostics, kernel logs)

### Use Cases:
- Communication pattern analysis
- Suspect contact identification
- Message content review
- Call timeline reconstruction
- Suspicious message detection

### Example Scenario:
```
Investigator needs to:
- Review suspect's text messages
- Analyze call patterns
- Identify contacts
- Detect suspicious communications

→ LEGAL consent is required
```

---

## 🔴 LEVEL 3: FULL

**Value**: 3
**Color**: Red 🔴
**Purpose**: Comprehensive forensic analysis

### What's Included:
- ✅ Everything from STANDARD level
- ✅ Everything from LEGAL level
- ✅ System Logs (system diagnostics, kernel logs, debug info)
- ✅ System Configuration (system settings, installed apps, services)
- ✅ Advanced Diagnostics (performance logs, crash reports, error logs)

### What's NOT Included:
- ❌ Nothing - FULL access to all data

### Use Cases:
- Complete device forensics
- System-level investigation
- Advanced threat analysis
- Comprehensive evidence collection
- Legal proceedings requiring complete data

### Example Scenario:
```
Investigator needs to:
- Complete forensic analysis
- System-level investigation
- All available device data
- Legal proceedings evidence

→ FULL consent is required
```

---

## 📊 CONSENT LEVEL COMPARISON

| Feature | STANDARD | LEGAL | FULL |
|---------|----------|-------|------|
| Device Info | ✅ | ✅ | ✅ |
| Location | ✅ | ✅ | ✅ |
| Media | ✅ | ✅ | ✅ |
| Security | ✅ | ✅ | ✅ |
| Communications | ❌ | ✅ | ✅ |
| System Logs | ❌ | ❌ | ✅ |
| **Access Level** | **Basic** | **Legal** | **Full** |

---

## 🔒 MODULE ACCESS BY CONSENT LEVEL

### Device Info Module
```
Required Level: STANDARD
Accessible With: STANDARD, LEGAL, FULL
```

### Location Module
```
Required Level: STANDARD
Accessible With: STANDARD, LEGAL, FULL
```

### Media Module
```
Required Level: STANDARD
Accessible With: STANDARD, LEGAL, FULL
```

### Security Module
```
Required Level: STANDARD
Accessible With: STANDARD, LEGAL, FULL
```

### Communications Module
```
Required Level: LEGAL
Accessible With: LEGAL, FULL
NOT Accessible With: STANDARD
```

### System Module
```
Required Level: FULL
Accessible With: FULL
NOT Accessible With: STANDARD, LEGAL
```

---

## 🎯 CONSENT HIERARCHY

```
FULL (3)
  ↑
  Includes all LEGAL features
  ↑
LEGAL (2)
  ↑
  Includes all STANDARD features
  ↑
STANDARD (1)
  ↑
  Base level
```

**Rule**: Higher consent level includes all features from lower levels.

---

## 💡 CHOOSING THE RIGHT CONSENT LEVEL

### Choose STANDARD When:
- You need basic device information
- You're tracking location patterns
- You need to review media files
- You're assessing device security
- Initial investigation phase

### Choose LEGAL When:
- You need to analyze communications
- You're investigating suspicious messages
- You need call records
- You're building communication patterns
- Legal investigation phase

### Choose FULL When:
- You need complete device forensics
- You're conducting system-level investigation
- You need all available data
- Legal proceedings require comprehensive evidence
- Final investigation phase

---

## 🔐 SECURITY IMPLICATIONS

### STANDARD Level
- **Risk**: Low
- **Privacy Impact**: Medium
- **Data Sensitivity**: Medium
- **Typical Approval**: Easier to obtain

### LEGAL Level
- **Risk**: Medium
- **Privacy Impact**: High
- **Data Sensitivity**: High
- **Typical Approval**: Requires legal justification

### FULL Level
- **Risk**: High
- **Privacy Impact**: Very High
- **Data Sensitivity**: Very High
- **Typical Approval**: Requires strong legal justification

---

## 📋 CONSENT FORM DISPLAY

When nominee approves, they see:

```
🟡 STANDARD
Device + Location + Media + Security

🟠 LEGAL
All data including Communications

🔴 FULL
Complete access including System logs
```

---

## 🧪 TESTING WITH CONSENT LEVELS

### Auto-Approve (Testing)
```python
# Auto-approve with LEGAL consent
ConsentTestingLoopholes.auto_approve_consent(
    consent_manager, 
    'CASE-001', 
    'LEGAL'
)
```

### Mock Consent (Testing)
```python
# Create mock FULL consent
ConsentTestingLoopholes.create_mock_consent(
    consent_manager,
    'CASE-001',
    'FULL'
)
```

### Reset (Testing)
```python
# Reset consent for testing
ConsentTestingLoopholes.reset_case_consent(
    consent_manager,
    'CASE-001'
)
```

---

## 📊 EXTRACTION BEHAVIOR BY CONSENT LEVEL

### With STANDARD Consent:
```
✅ Device Info Extraction    - SUCCESS
✅ Location Extraction       - SUCCESS
✅ Media Extraction          - SUCCESS
✅ Security Extraction       - SUCCESS
❌ Communications Extraction - BLOCKED
❌ System Extraction         - BLOCKED
```

### With LEGAL Consent:
```
✅ Device Info Extraction    - SUCCESS
✅ Location Extraction       - SUCCESS
✅ Media Extraction          - SUCCESS
✅ Security Extraction       - SUCCESS
✅ Communications Extraction - SUCCESS
❌ System Extraction         - BLOCKED
```

### With FULL Consent:
```
✅ Device Info Extraction    - SUCCESS
✅ Location Extraction       - SUCCESS
✅ Media Extraction          - SUCCESS
✅ Security Extraction       - SUCCESS
✅ Communications Extraction - SUCCESS
✅ System Extraction         - SUCCESS
```

---

## 🔄 CONSENT LEVEL PROGRESSION

Typical investigation flow:

```
1. STANDARD Consent
   ↓
   Initial analysis
   ↓
2. LEGAL Consent (if needed)
   ↓
   Communication analysis
   ↓
3. FULL Consent (if needed)
   ↓
   Complete forensics
```

---

## ⚠️ IMPORTANT NOTES

1. **Immutable**: Once set, consent level cannot be changed (must revoke & create new)
2. **Hierarchical**: Higher levels include all lower level features
3. **Audit Trail**: All consent events are logged
4. **Revocation**: Consent can be revoked at any time
5. **Testing**: Use testing loopholes only in development mode

---

## 🎯 SUMMARY

**3 Consent Levels:**

| Level | Name | Access | Use Case |
|-------|------|--------|----------|
| 1 | STANDARD | Device + Location + Media + Security | Basic forensics |
| 2 | LEGAL | + Communications | Legal investigation |
| 3 | FULL | + System Logs | Complete forensics |

**Key Principle**: Higher consent = More data access = Higher privacy impact

---

## ✅ IMPLEMENTATION STATUS

- ✅ 3 Consent Levels defined
- ✅ Module minimum levels configured
- ✅ Consent form updated
- ✅ Testing loopholes updated
- ✅ Documentation complete

**Ready for PHASE 2 (Extraction Module)**
