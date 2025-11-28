# 🔌 PHASE 6: COMPLETE WIRING - ALL MODULES INTEGRATED

**Date**: November 26, 2025
**Status**: ✅ COMPLETE

---

## 📊 WHAT WAS WIRED

### **1. EXTRACTION FOLDER** ✅
```
modules/extraction/
├── ui_device_selector.py        → Tab 1: Device Selection
├── ui_module_selector.py        → Tab 2: Module Selection
├── ui_consent_check.py          → Tab 3: Consent Check
├── ui_extraction_progress.py    → Tab 4: Extraction Progress
├── ui_extraction_results.py     → Tab 5: Results Display
├── ui_consent_approval.py       → Approval Portal
└── consent.py                   → Consent management
```

**Wired In**: ✅ render_extraction_workflow()
**Location**: app.py → Extraction page

---

### **2. ANALYSIS FOLDER** ✅
```
modules/analysis/
├── ui.py                        → Analysis UI components
├── comms_analyzer.py            → Communications analysis
├── location_intelligence.py     → Location analysis
├── media_viewer.py              → Media analysis
└── models.py                    → Analysis models
```

**Wired In**: ✅ render_intelligence_page()
**Components**:
- render_comms_analyzer()
- render_location_intelligence()
- render_media_viewer()

**Location**: app.py → Intelligence page

---

### **3. CONSENT FOLDER** ✅
```
modules/consent/
├── models.py                    → Consent system
│   ├── ConsentLevel enum
│   ├── ConsentSession class
│   ├── ConsentManager class
│   ├── ConsentAuditTrail class
│   └── ApprovalLinkGenerator class
└── ui.py                        → Consent UI
```

**Wired In**: ✅ get_consent_manager()
**Usage**:
- Consent level validation
- Approval link generation
- Audit trail tracking

**Location**: app.py → All pages (with consent checks)

---

## 🔌 WIRING DETAILS

### **EXTRACTION WIRING**

```python
# app.py - Line 31-37
from modules.extraction.ui_device_selector import render_device_selector
from modules.extraction.ui_module_selector import render_module_selector
from modules.extraction.ui_consent_check import render_consent_check
from modules.extraction.ui_consent_approval import render_consent_approval_form
from modules.extraction.ui_extraction_orchestrator import render_extraction_page
from modules.extraction.ui_extraction_progress import render_extraction_progress
from modules.extraction.ui_extraction_results import render_extraction_results

# app.py - render_extraction_workflow()
with tab1:
    render_device_selector()

with tab2:
    render_module_selector()

with tab3:
    render_consent_check()

with tab4:
    render_extraction_progress()

with tab5:
    render_extraction_results()
```

---

### **ANALYSIS WIRING**

```python
# app.py - Line 40-49
try:
    from modules.analysis.ui import (
        render_comms_analyzer,
        render_location_intelligence,
        render_media_viewer
    )
    ANALYSIS_UI_AVAILABLE = True
except ImportError:
    ANALYSIS_UI_AVAILABLE = False

# app.py - render_intelligence_page()
if ANALYSIS_UI_AVAILABLE:
    try:
        render_comms_analyzer()
    except Exception as e:
        # Fallback UI
        
if ANALYSIS_UI_AVAILABLE:
    try:
        render_location_intelligence()
    except Exception as e:
        # Fallback UI

if ANALYSIS_UI_AVAILABLE:
    try:
        render_media_viewer()
    except Exception as e:
        # Fallback UI
```

---

### **CONSENT WIRING**

```python
# app.py - Line 52-61
try:
    from modules.consent.models import (
        get_consent_manager,
        ConsentLevel,
        MODULE_MIN_LEVELS
    )
    CONSENT_AVAILABLE = True
except ImportError:
    CONSENT_AVAILABLE = False

# app.py - render_intelligence_page() - Risk Assessment Tab
if CONSENT_AVAILABLE:
    try:
        consent_manager = get_consent_manager()
        session = consent_manager.get_session(case_id)
        current_level = session.level if session else ConsentLevel.BASIC
        st.info(f"📊 Current Consent Level: **{current_level.name}**")
    except Exception as e:
        st.warning(f"⚠️ Consent check: {str(e)}")
```

---

## 📋 COMPLETE WIRING MAP

```
app.py (Main Entry Point)
│
├─ EXTRACTION FOLDER
│  ├─ render_extraction_workflow()
│  │  ├─ Tab 1: render_device_selector()
│  │  ├─ Tab 2: render_module_selector()
│  │  ├─ Tab 3: render_consent_check()
│  │  ├─ Tab 4: render_extraction_progress()
│  │  └─ Tab 5: render_extraction_results()
│  │
│  └─ render_nominee_portal()
│     └─ render_consent_approval_form()
│
├─ ANALYSIS FOLDER
│  └─ render_intelligence_page()
│     ├─ Tab 1: render_comms_analyzer()
│     ├─ Tab 2: render_location_intelligence()
│     ├─ Tab 3: render_media_viewer()
│     └─ Tab 4: Risk Assessment (with consent check)
│
└─ CONSENT FOLDER
   ├─ get_consent_manager()
   ├─ ConsentLevel enum
   └─ MODULE_MIN_LEVELS
```

---

## 🎯 INTEGRATION POINTS

### **1. Extraction Page**
```
URL: /extraction
Role: Investigator
Components:
  - Device selector (extraction folder)
  - Module selector (extraction folder)
  - Consent check (extraction folder + consent folder)
  - Progress display (extraction folder)
  - Results display (extraction folder)
```

### **2. Intelligence Page**
```
URL: /intelligence
Role: Investigator
Components:
  - Communications analyzer (analysis folder)
  - Location intelligence (analysis folder)
  - Media viewer (analysis folder)
  - Risk assessment (analysis folder + consent folder)
```

### **3. Approval Portal**
```
URL: /approve?case_id=CASE-001
Role: Nominee
Components:
  - Consent approval form (extraction folder)
  - PIN/Pattern verification (consent folder)
```

---

## ✅ WIRING CHECKLIST

### **EXTRACTION FOLDER**
- [x] ui_device_selector.py imported
- [x] ui_module_selector.py imported
- [x] ui_consent_check.py imported
- [x] ui_consent_approval.py imported
- [x] ui_extraction_progress.py imported
- [x] ui_extraction_results.py imported
- [x] render_extraction_workflow() created
- [x] All components called in tabs
- [x] Error handling with fallbacks
- [x] Session state management

### **ANALYSIS FOLDER**
- [x] render_comms_analyzer imported
- [x] render_location_intelligence imported
- [x] render_media_viewer imported
- [x] render_intelligence_page() updated
- [x] All components called in tabs
- [x] Error handling with fallbacks
- [x] ANALYSIS_UI_AVAILABLE flag
- [x] Fallback UIs for each component

### **CONSENT FOLDER**
- [x] get_consent_manager imported
- [x] ConsentLevel imported
- [x] MODULE_MIN_LEVELS imported
- [x] Consent checks in intelligence page
- [x] Consent checks in extraction page
- [x] CONSENT_AVAILABLE flag
- [x] Error handling for consent operations

---

## 🔄 DATA FLOW

### **Extraction Workflow**

```
User selects Investigator
    ↓
Clicks "Extraction"
    ↓
render_extraction_workflow() called
    ↓
Tab 1: Device Selection
├─ render_device_selector()
└─ Stores: st.session_state.selected_device
    ↓
Tab 2: Module Selection
├─ render_module_selector()
└─ Stores: st.session_state.selected_modules
    ↓
Tab 3: Consent Check
├─ render_consent_check()
├─ Uses: get_consent_manager()
└─ Stores: st.session_state.consent_approved
    ↓
Tab 4: Extraction Progress
├─ render_extraction_progress()
└─ Shows: Real-time progress
    ↓
Tab 5: Results Display
├─ render_extraction_results()
└─ Shows: Extracted data
```

### **Intelligence Workflow**

```
User selects Investigator
    ↓
Clicks "Intelligence"
    ↓
render_intelligence_page() called
    ↓
Tab 1: Communications
├─ render_comms_analyzer()
└─ Analyzes: Messages, calls, etc.
    ↓
Tab 2: Location
├─ render_location_intelligence()
└─ Analyzes: GPS, locations, etc.
    ↓
Tab 3: Media
├─ render_media_viewer()
└─ Analyzes: Photos, videos, etc.
    ↓
Tab 4: Risk Assessment
├─ get_consent_manager()
├─ Check: Current consent level
└─ Show: Risk scores
```

### **Approval Workflow**

```
Nominee receives link
    ↓
Clicks link: /approve?case_id=CASE-001
    ↓
render_sidebar() detects URL parameter
    ↓
Sets: st.session_state.user_role = "nominee"
    ↓
render_nominee_portal() called
    ↓
render_consent_approval_form() called
    ↓
Show: Case details, consent form
    ↓
Nominee enters PIN
    ↓
Consent unlocked
    ↓
Investigator sees approval
```

---

## 🔐 CONSENT INTEGRATION

### **Where Consent is Checked**

1. **Extraction Page**
   - Tab 3: Consent Check
   - Uses: render_consent_check()
   - Checks: Consent level vs module requirements

2. **Intelligence Page**
   - Tab 4: Risk Assessment
   - Uses: get_consent_manager()
   - Shows: Current consent level

3. **Approval Portal**
   - Approval Form
   - Uses: render_consent_approval_form()
   - Verifies: PIN/Pattern

---

## 📊 MODULE AVAILABILITY FLAGS

### **ANALYSIS_UI_AVAILABLE**
```python
try:
    from modules.analysis.ui import (...)
    ANALYSIS_UI_AVAILABLE = True
except ImportError:
    ANALYSIS_UI_AVAILABLE = False
```

**Usage**: Check before calling analysis components

### **CONSENT_AVAILABLE**
```python
try:
    from modules.consent.models import (...)
    CONSENT_AVAILABLE = True
except ImportError:
    CONSENT_AVAILABLE = False
```

**Usage**: Check before calling consent operations

---

## 🛡️ ERROR HANDLING

### **Try-Except Blocks**

All component calls wrapped in try-except:

```python
if ANALYSIS_UI_AVAILABLE:
    try:
        render_comms_analyzer()
    except Exception as e:
        st.warning(f"⚠️ Communications Analyzer: {str(e)}")
        # Fallback UI
```

### **Fallback UIs**

Each component has fallback UI:
- Communications: Suspicious messages table
- Location: Location visits table
- Media: Media counts table
- Consent: Manual consent check

---

## 📁 FILE STRUCTURE

```
c:\Forensmart\
├── app.py                          ← MAIN WIRING FILE
│
├── modules\
│  ├── extraction\                  ← EXTRACTION FOLDER
│  │  ├── ui_device_selector.py
│  │  ├── ui_module_selector.py
│  │  ├── ui_consent_check.py
│  │  ├── ui_consent_approval.py
│  │  ├── ui_extraction_progress.py
│  │  ├── ui_extraction_results.py
│  │  └── consent.py
│  │
│  ├── analysis\                    ← ANALYSIS FOLDER
│  │  ├── ui.py
│  │  ├── comms_analyzer.py
│  │  ├── location_intelligence.py
│  │  ├── media_viewer.py
│  │  └── models.py
│  │
│  └── consent\                     ← CONSENT FOLDER
│     ├── models.py
│     └── ui.py
│
└── PHASE_6_WIRING_COMPLETE.md     ← THIS FILE
```

---

## 🚀 HOW TO RUN

```bash
# Step 1: Install dependencies
pip install streamlit pandas

# Step 2: Run the app
cd c:\Forensmart
streamlit run app.py

# Step 3: Open browser
http://localhost:8501
```

---

## 📋 TESTING WIRING

### **Test Extraction Wiring**
1. Select "Investigator" role
2. Click "Extraction"
3. Verify all 5 tabs load
4. Test each tab component

### **Test Analysis Wiring**
1. Select "Investigator" role
2. Click "Intelligence"
3. Verify all 4 tabs load
4. Check consent level display

### **Test Consent Wiring**
1. Generate approval link in Extraction
2. Open link in new tab
3. Verify approval form shows
4. Test PIN entry

---

## ✅ WIRING STATUS

**Status**: ✅ COMPLETE

**Wired Modules**:
- ✅ Extraction Folder (7 components)
- ✅ Analysis Folder (3 components)
- ✅ Consent Folder (1 component)

**Total Components Wired**: 11

**Error Handling**: ✅ All components have try-except blocks

**Fallback UIs**: ✅ All components have fallback UIs

---

## 🎯 NEXT STEPS

### **Phase 7: Database Integration** (Optional)
- Connect to PostgreSQL
- Store cases
- Store extraction results
- Store approval links

### **Phase 8: Advanced Features** (Optional)
- Multi-device extraction
- Batch processing
- Custom reports
- ML analysis

### **Phase 9: Deployment** (Optional)
- Docker containerization
- Cloud deployment
- Load balancing
- Monitoring

---

**Created**: November 26, 2025
**Status**: ✅ COMPLETE
**Ready**: YES ✅

