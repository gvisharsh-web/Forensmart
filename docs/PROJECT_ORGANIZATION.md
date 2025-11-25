# ForenSmart Project Organization Guide

## Overview

ForenSmart has been reorganized into a professional, scalable project structure following Python best practices.

## Directory Structure

```
ForenSmart/
├── 📄 README.md                    # Project overview
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
├── 📄 pyproject.toml              # Project configuration
├── 🚀 app.py                       # Main entry point
│
├── 📁 modules/                     # All application code
│   ├── __init__.py
│   │
│   ├── approval/                   # Approval system
│   │   ├── __init__.py
│   │   ├── manager.py             # Unified approval manager
│   │   ├── sync.py                # Approval synchronization
│   │   ├── utils.py               # Approval utilities
│   │   ├── redirect.py            # Redirect handling
│   │   └── auto_extraction.py     # Auto-extraction trigger
│   │
│   ├── consent/                    # Consent management
│   │   ├── __init__.py
│   │   ├── models.py              # ConsentLevel, ConsentSession
│   │   ├── manager.py             # Consent manager
│   │   ├── portal.py              # Consent portal UI
│   │   └── enhanced.py            # Enhanced features
│   │
│   ├── extraction/                 # Data extraction
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Main orchestrator
│   │   ├── validator.py           # Pre-extraction checks
│   │   ├── progress.py            # Progress tracking
│   │   └── ui.py                  # Extraction UI
│   │
│   ├── analysis/                   # Data analysis
│   │   ├── __init__.py
│   │   ├── comms_analyzer.py      # Communications analysis
│   │   ├── location_intelligence.py # Location analysis
│   │   └── suspicious_classifier.py # Suspicious detection
│   │
│   ├── storage/                    # Storage management
│   │   ├── __init__.py
│   │   ├── manager.py             # Storage operations
│   │   └── ui.py                  # Storage UI
│   │
│   ├── ui/                         # UI components
│   │   ├── __init__.py
│   │   ├── progress_ui.py         # Progress display
│   │   ├── media_viewer.py        # Media viewing
│   │   └── suspicious_comms_ui.py # Suspicious comms UI
│   │
│   ├── adapters/                   # Device adapters
│   │   ├── __init__.py
│   │   ├── android_adb.py         # Android ADB adapter
│   │   ├── ios_logical.py         # iOS adapter
│   │   ├── hdd_imager.py          # HDD adapter
│   │   └── interface.py           # Adapter interface
│   │
│   ├── automation/                 # Automation features (NEW)
│   │   ├── __init__.py
│   │   ├── scheduler.py           # Task scheduler
│   │   └── workflow.py            # Workflow engine
│   │
│   ├── reporting/                  # Report generation (NEW)
│   │   ├── __init__.py
│   │   ├── ai_generator.py        # AI report generator
│   │   └── templates.py           # Report templates
│   │
│   └── shared/                     # Shared utilities
│       ├── __init__.py
│       ├── utils.py               # Common utilities
│       ├── error_checker.py       # Error checking
│       ├── device_manager.py      # Device management
│       ├── device_detector.py     # Device detection
│       ├── file_handler.py        # File operations
│       ├── unified_error_system.py # Error system
│       └── app_error_checker.py   # App error checking
│
├── 📁 pages/                       # Streamlit pages
│   ├── 01_consent_portal.py       # Consent portal page
│   ├── 02_extraction.py           # Extraction page
│   ├── 03_intelligence.py         # Intelligence page
│   ├── 04_reports_storage.py      # Reports & storage page
│   ├── 05_diagnostics.py          # Diagnostics page
│   └── 06_automation_reports.py   # Automation & reports (NEW)
│
├── 📁 data/                        # Runtime data (gitignored)
│   ├── artifacts/                 # Extracted artifacts
│   ├── audit/                     # Audit logs
│   ├── consent_records/           # Consent records
│   ├── case_snapshots/            # Case snapshots
│   └── reports/                   # Generated reports
│
├── 📁 docs/                        # Documentation
│   ├── PROJECT_ORGANIZATION.md    # This file
│   ├── ARCHITECTURE.md            # Architecture guide
│   ├── SETUP.md                   # Setup instructions
│   ├── API.md                     # API documentation
│   ├── FEATURES.md                # Feature list
│   └── (80+ other docs)           # Historical documentation
│
├── 📁 scripts/                     # Utility scripts
│   └── fix_forensmart_lint.py     # Linting fixer
│
├── 📁 tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_approval.py           # Approval tests
│   ├── test_consent.py            # Consent tests
│   ├── test_extraction.py         # Extraction tests
│   └── conftest.py                # Pytest configuration
│
├── 📁 .backups/                    # Old files (gitignored)
│   ├── app_patched_fixed_final.py
│   ├── app_patched_fixed_noconsent.py
│   └── (old versions)
│
├── 📁 driver_bundle/               # Device drivers
│   └── (driver files)
│
├── 🔧 .env.example                # Environment variables template
├── 🔧 .gitignore                  # Git ignore rules
└── 🔧 .github/                    # GitHub workflows
```

## Module Organization

### approval/
Handles all approval-related operations:
- **manager.py**: Unified approval manager (single source of truth)
- **sync.py**: Synchronization with file-based approval
- **utils.py**: Utility functions for approval
- **redirect.py**: Redirect handling after approval
- **auto_extraction.py**: Automatic extraction trigger

### consent/
Manages consent levels and sessions:
- **models.py**: ConsentLevel enum, ConsentSession dataclass
- **manager.py**: Consent manager for sessions
- **portal.py**: Consent portal UI
- **enhanced.py**: Enhanced consent features

### extraction/
Core data extraction functionality:
- **orchestrator.py**: Main extraction orchestrator
- **validator.py**: Pre-extraction validation
- **progress.py**: Progress tracking
- **ui.py**: Extraction UI components

### analysis/
Data analysis and intelligence:
- **comms_analyzer.py**: Communications analysis
- **location_intelligence.py**: Location analysis
- **suspicious_classifier.py**: Suspicious activity detection

### storage/
Storage management:
- **manager.py**: Storage operations (delete, cleanup)
- **ui.py**: Storage management UI

### ui/
Streamlit UI components:
- **progress_ui.py**: Progress bar display
- **media_viewer.py**: Media viewing
- **suspicious_comms_ui.py**: Suspicious communications UI

### adapters/
Device communication adapters:
- **android_adb.py**: Android ADB adapter
- **ios_logical.py**: iOS logical extraction
- **hdd_imager.py**: HDD imaging
- **interface.py**: Adapter interface

### automation/ (NEW)
Automation features:
- **scheduler.py**: Task scheduling
- **workflow.py**: Workflow engine

### reporting/ (NEW)
Report generation:
- **ai_generator.py**: AI-powered report generation
- **templates.py**: Report templates

### shared/
Shared utilities:
- **utils.py**: Common utility functions
- **error_checker.py**: Error checking
- **device_manager.py**: Device management
- **device_detector.py**: Device detection
- **file_handler.py**: File operations
- **unified_error_system.py**: Error system
- **app_error_checker.py**: App error checking

## Import Examples

### Before Reorganization
```python
from modules.approval_manager import ApprovalManager
from modules.consent import ConsentManager
from modules.data_extraction_orchestrator import DataExtractionOrchestrator
```

### After Reorganization
```python
from modules.approval.manager import ApprovalManager
from modules.consent.manager import ConsentManager
from modules.extraction.orchestrator import DataExtractionOrchestrator
```

## Data Flow

```
1. User creates case
   └─> modules/consent/portal.py

2. Nominee approves
   └─> modules/approval/manager.py

3. Extraction starts
   └─> modules/extraction/orchestrator.py
       ├─> modules/adapters/android_adb.py
       ├─> modules/adapters/ios_logical.py
       └─> modules/adapters/hdd_imager.py

4. Data analyzed
   ├─> modules/analysis/comms_analyzer.py
   ├─> modules/analysis/location_intelligence.py
   └─> modules/analysis/suspicious_classifier.py

5. Report generated
   └─> modules/reporting/ai_generator.py

6. Results stored
   └─> modules/storage/manager.py
```

## Key Components

### ConsentManager
- Manages consent levels (NONE, BASIC, STANDARD, LEGAL)
- Tracks consent sessions
- Provides consent validation

### ApprovalManager
- Unified approval handling
- Supports multiple approval sources
- Fallback strategy (online → offline)

### DataExtractionOrchestrator
- Coordinates all extraction modules
- Manages progress callbacks
- Handles error recovery

### AutomationScheduler (NEW)
- Schedules periodic tasks
- Manages background jobs
- Supports multiple job types

### AIReportGenerator (NEW)
- Generates professional reports
- Uses ChatGPT/Claude
- Exports to PDF/TXT

## Best Practices

### Imports
- Always use absolute imports: `from modules.approval.manager import ApprovalManager`
- Never use relative imports in production code
- Group imports: stdlib, third-party, local

### Module Organization
- Keep modules focused on single responsibility
- Use `__init__.py` to expose public APIs
- Keep internal functions private (prefix with `_`)

### Error Handling
- Always log errors with context
- Use specific exception types
- Provide user-friendly error messages

### Testing
- Write tests in `tests/` directory
- Use pytest for test framework
- Mock external dependencies

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

# Run tests
pytest tests/

# Check code quality
ruff check .
```

## Git Workflow

```bash
# After reorganization
git add .
git commit -m "refactor: reorganize project structure"
git push origin main
```

## Migration Notes

- All data folders moved to `data/`
- All documentation moved to `docs/`
- Old app files backed up in `.backups/`
- All imports updated automatically
- No functionality changed, only organization

## Future Improvements

1. Add API module for REST endpoints
2. Add database module for data persistence
3. Add logging module for centralized logging
4. Add config module for configuration management
5. Add middleware module for request handling

## Questions?

Refer to `docs/ARCHITECTURE.md` for detailed architecture information.
