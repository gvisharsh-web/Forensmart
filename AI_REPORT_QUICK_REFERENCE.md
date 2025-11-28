# 🤖 AI REPORT GENERATION - QUICK REFERENCE

**Status**: ✅ IMPLEMENTED

---

## 📊 REPORT TYPES AT A GLANCE

| Report Type | Pages | Audience | Purpose |
|-------------|-------|----------|---------|
| **Executive Summary** | 1-2 | Managers, Legal | High-level overview |
| **Detailed Findings** | 5-10 | Investigators | In-depth analysis |
| **Technical Analysis** | 3-5 | Experts | Methodology & specs |
| **Risk Assessment** | 2-3 | Risk Managers | Risk identification |
| **Timeline Report** | 3-5 | Prosecutors | Chronological view |
| **Full Report** | 15-25 | Legal Teams | Complete documentation |

---

## 🚀 QUICK START

### **Step 1: Import**
```python
from modules.shared.ai_report_generator import AIReportGenerator, ReportExporter
```

### **Step 2: Initialize**
```python
generator = AIReportGenerator(
    case_id="CASE-001",
    case_details={
        'investigator': 'John Smith',
        'nominee_name': 'Jane Doe',
        'device_type': 'Android',
        'reason': 'Criminal Investigation',
        'consent_level': 'LEGAL'
    }
)
```

### **Step 3: Generate**
```python
report = generator.generate_executive_summary(extraction_results)
```

### **Step 4: Export**
```python
ReportExporter.export_to_text(report, "report.txt")
ReportExporter.export_to_json(data, "data.json")
ReportExporter.export_to_pdf(report, "report.pdf")
```

---

## 📋 REPORT CONTENTS

### **Executive Summary**
```
✓ Case information
✓ Investigation overview
✓ Extraction summary
✓ Key findings
✓ Risk assessment
✓ Next steps
```

### **Detailed Findings**
```
✓ Communications analysis
✓ Location intelligence
✓ Media analysis
✓ Device information
✓ Security findings
✓ Evidence summary
```

### **Technical Analysis**
```
✓ Extraction methodology
✓ Device specifications
✓ Data extraction details
✓ Chain of custody
✓ Quality metrics
```

### **Risk Assessment**
```
✓ Overall risk level
✓ Risk breakdown
✓ Critical findings
✓ High priority items
✓ Recommendations
✓ Investigation priorities
```

### **Timeline Report**
```
✓ Chronological events
✓ Communication timeline
✓ Location timeline
✓ Media timeline
```

### **Full Report**
```
✓ All sections combined
✓ Report certification
✓ Professional formatting
```

---

## 💾 EXPORT FORMATS

### **Text (.txt)**
```python
ReportExporter.export_to_text(report, "report.txt")
```
- Plain text
- Universal compatibility
- Easy to read

### **JSON (.json)**
```python
ReportExporter.export_to_json(data, "data.json")
```
- Structured data
- Machine-readable
- Integration-friendly

### **PDF (.pdf)**
```python
ReportExporter.export_to_pdf(report, "report.pdf")
```
- Professional appearance
- Print-ready
- Requires reportlab

---

## 🎯 USAGE EXAMPLES

### **Generate Executive Summary**
```python
summary = generator.generate_executive_summary(extraction_results)
print(summary)
```

### **Generate Detailed Findings**
```python
findings = generator.generate_detailed_findings(extraction_results)
print(findings)
```

### **Generate Technical Analysis**
```python
technical = generator.generate_technical_analysis(extraction_results)
print(technical)
```

### **Generate Risk Assessment**
```python
risk = generator.generate_risk_assessment(extraction_results)
print(risk)
```

### **Generate Timeline Report**
```python
timeline = generator.generate_timeline_report(extraction_results)
print(timeline)
```

### **Generate Full Report**
```python
full = generator.generate_full_report(extraction_results)
print(full)
```

---

## 📊 EXTRACTION RESULTS FORMAT

```python
extraction_results = {
    # Size information
    'total_size': 48_000_000_000,  # bytes
    
    # Counts
    'file_count': 12450,
    'message_count': 3245,
    'media_count': 8932,
    'location_count': 127,
    
    # Risk assessment
    'critical_count': 15,
    'high_count': 42,
    'medium_count': 85,
    'risk_score': 65,
    
    # Communications
    'sms_count': 1200,
    'email_count': 450,
    'chat_app_count': 3,
    'suspicious_messages': 42,
    'top_contacts': [
        {'name': 'Contact 1', 'message_count': 250},
        {'name': 'Contact 2', 'message_count': 180}
    ],
    
    # Location
    'gps_count': 5000,
    'frequent_locations': [
        {'name': 'Location A', 'visits': 45},
        {'name': 'Location B', 'visits': 32}
    ],
    
    # Media
    'photo_count': 5000,
    'video_count': 2500,
    'audio_count': 1432,
    
    # Device info
    'device_model': 'Samsung Galaxy S21',
    'os_version': 'Android 12',
    'last_boot': '2025-11-26 08:15:00',
    'storage_used': 45_000_000_000,
    'storage_available': 15_000_000_000,
    
    # Security
    'app_count': 150,
    'suspicious_apps': 3,
    'malware_count': 0,
    'security_issues': 2,
    
    # Evidence
    'evidence_count': 500,
    'critical_evidence': 15,
    'supporting_evidence': 485,
    
    # Extraction details
    'extraction_method': 'Physical extraction',
    'extraction_duration': '45 minutes',
    'integrity_status': 'VERIFIED',
    'hash_verified': 'YES'
}
```

---

## 🔧 HELPER METHODS

### **Format Size**
```python
AIReportGenerator._format_size(1_000_000_000)
# Output: "1.00 GB"
```

### **Assess Risk Level**
```python
AIReportGenerator._assess_risk_level({'risk_score': 75})
# Output: "HIGH"
```

### **Format Contacts**
```python
AIReportGenerator._format_contacts([
    {'name': 'John', 'message_count': 100},
    {'name': 'Jane', 'message_count': 80}
])
# Output: Formatted contact list
```

---

## 📁 FILE LOCATION

**Module**: `c:\Forensmart\modules\shared\ai_report_generator.py`

**Classes**:
- `AIReportGenerator` - Main report generator
- `ReportExporter` - Export functionality

**Methods**:
- `generate_executive_summary()`
- `generate_detailed_findings()`
- `generate_technical_analysis()`
- `generate_risk_assessment()`
- `generate_timeline_report()`
- `generate_full_report()`

---

## ✅ FEATURES

✅ 6 report types
✅ 3 export formats
✅ Human-readable output
✅ Professional formatting
✅ Automated generation
✅ Customizable content
✅ Chain of custody
✅ Risk assessment
✅ Timeline analysis
✅ Evidence linking

---

## 🎯 NEXT STEPS

1. **Integration**: Add to app.py reports page
2. **UI**: Create report generation UI
3. **Export**: Add export buttons
4. **Scheduling**: Add scheduled reports
5. **Templates**: Add custom templates

---

**Status**: ✅ READY FOR USE
**Location**: `modules/shared/ai_report_generator.py`
**Documentation**: `AI_REPORT_GENERATION_SYSTEM.md`

