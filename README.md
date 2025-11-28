# 🔍 FORENSMART - Advanced Digital Forensics Platform

**Enterprise-Grade Forensic Analysis with AI-Powered Intelligence**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Forensmart** is a comprehensive digital forensics platform that combines local forensic analysis with cloud-based intelligence. It provides:

- **Unified Consent Management** - Immutable consent levels with audit trails
- **Multi-Module Extraction** - Device info, communications, location, media, security, system data
- **AI-Powered Analysis** - Suspicious message detection, pattern recognition, risk scoring
- **Cloud Integration** - AWS S3 storage, bidirectional sync, cloud analysis APIs
- **Real-time Collaboration** - Investigator dashboard + nominee approval portal
- **Automation Engine** - Scheduled extraction, workflow automation, notifications
- **AI Report Generation** - Intelligent summaries, evidence correlation, professional formatting
- **Self-Healing System** - Automatic error detection and fixing, silent error detection

---

## ✨ Key Features

### 🔐 Consent Management
- ✅ Immutable consent levels (NONE, BASIC, STANDARD, LEGAL, FULL)
- ✅ Centralized consent enforcement
- ✅ Audit trail logging
- ✅ Approval link generation & sharing
- ✅ Real-time approval synchronization

### 📱 Data Extraction
- ✅ Device information extraction
- ✅ Communications (SMS, calls, contacts, messaging)
- ✅ Location data (GPS, cell towers)
- ✅ Security settings & authentication
- ✅ Media files & thumbnails
- ✅ System logs & diagnostics

### 🧠 Intelligence & Analysis
- ✅ Suspicious message classification (TF-IDF model)
- ✅ Communication pattern analysis
- ✅ Location clustering & visualization
- ✅ Media analysis & categorization
- ✅ Risk scoring & threat assessment
- ✅ Evidence correlation

### ☁️ Cloud Integration
- ✅ AWS S3 storage & backup
- ✅ Cloud API calls for analysis
- ✅ Bidirectional sync (local ↔ cloud)
- ✅ Automatic backup & recovery
- ✅ Offline support with queuing

### 🤖 Automation & AI
- ✅ Scheduled extraction workflows
- ✅ Automatic analysis & reporting
- ✅ Notification system
- ✅ AI-powered report generation
- ✅ Multi-language support

### 🔧 Self-Healing System
- ✅ Automatic error detection & fixing
- ✅ Silent error detection (consent, data flow, state, logic)
- ✅ Real-time monitoring
- ✅ Auto-recovery mechanisms
- ✅ Comprehensive logging

---

## 🏗️ Architecture

### Consolidated Structure (9 Core Modules)

```
modules/
├── shared/utils.py           # Advanced utilities (1000-1200 lines)
├── consent/models.py         # Consent system (600-800 lines)
├── extraction/orchestrator.py # Extraction engine (800-1000 lines)
├── analysis/processors.py    # Analysis pipeline (600-800 lines)
├── intelligence/engine.py    # Intelligence engine (500-700 lines)
├── cloud/integration.py      # Cloud integration (800-1000 lines)
├── automation/engine.py      # Automation engine (500-700 lines)
├── ai/report_generator.py    # AI reports (500-700 lines)
└── ui/components.py          # UI components (400-600 lines)

api/
└── routes/handlers.py        # API endpoints (300-500 lines)

app.py                        # Unified portal (1000-1200 lines)
```

**Total: ~8000-10000 lines, highly organized, zero redundancy**

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Git
- AWS account (for cloud features)
- PostgreSQL (optional, for database)

### Setup

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/forensmart.git
cd forensmart
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.template .env
# Edit .env with your configuration
```

5. **Run Application**
```bash
streamlit run app.py
```

---

## ⚡ Quick Start

### 1. Create a Case
```
1. Open Forensmart
2. Click "Create New Case"
3. Enter case details
4. Select device
```

### 2. Generate Approval Link
```
1. Click "Generate Approval Link"
2. Share link with nominee via email/SMS
3. Nominee opens link and approves
```

### 3. Run Extraction
```
1. Extraction starts automatically after approval
2. Monitor progress in real-time
3. View results in Intelligence module
```

### 4. Analyze Findings
```
1. View suspicious messages
2. Analyze location patterns
3. Review media files
4. Generate AI report
```

---

## ⚙️ Configuration

### Environment Variables

See `.env.template` for all available configuration options:

```bash
# Streamlit
STREAMLIT_SERVER_PORT=8501

# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Database
DATABASE_URL=postgresql://user:pass@localhost/forensmart

# AI/LLM
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Security
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret

# Features
FEATURE_CLOUD_INTEGRATION=true
FEATURE_AUTOMATION=true
FEATURE_AI_REPORTS=true
```

---

## 📖 Usage

### Investigator Dashboard

```python
# Start extraction
case = create_case(device_id, consent_level)
results = extract_data(case_id)

# Analyze findings
suspicious = analyze_communications(results)
locations = analyze_location(results)
media = analyze_media(results)

# Generate report
report = generate_ai_report(case_id)
```

### Nominee Approval Portal

```
1. Receive approval link
2. Open link in browser
3. Review consent form
4. Approve with PIN/Pattern/Biometric
5. Extraction starts automatically
```

---

## 🌐 Deployment

### Streamlit Cloud

1. **Push to GitHub**
```bash
git push origin main
```

2. **Deploy to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select repository & branch
   - Set main file: `app.py`
   - Click "Deploy"

3. **Configure Secrets**
   - Add environment variables in Streamlit Cloud settings
   - Configure AWS credentials
   - Setup API keys

### Docker Deployment

```bash
docker build -t forensmart .
docker run -p 8501:8501 forensmart
```

---

## 📚 API Documentation

### REST API Endpoints

```
# Cases
GET    /api/cases
POST   /api/cases
GET    /api/cases/{case_id}
PUT    /api/cases/{case_id}
DELETE /api/cases/{case_id}

# Extraction
POST   /api/extraction/start
GET    /api/extraction/{case_id}/status
GET    /api/extraction/{case_id}/results

# Approval
POST   /api/approval/generate-link
GET    /api/approval/{token}
POST   /api/approval/{token}/approve

# Analysis
GET    /api/analysis/{case_id}/suspicious
GET    /api/analysis/{case_id}/location
GET    /api/analysis/{case_id}/media

# Reports
POST   /api/reports/generate
GET    /api/reports/{case_id}
```

See `docs/API_DOCUMENTATION.md` for detailed API docs.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules tests/

# Run specific test
pytest tests/test_consent.py
```

---

## 🔍 Monitoring

### Silent Error Detection Dashboard

The system continuously monitors for:
- ✅ Extraction consent problems
- ✅ Data flow issues
- ✅ State mismatches
- ✅ Logic bugs
- ✅ Cache staleness

View monitoring dashboard in app: **Settings → Monitoring**

---

## 📝 Documentation

- `docs/ARCHITECTURE.md` - System architecture
- `docs/API_DOCUMENTATION.md` - API reference
- `docs/USER_GUIDE.md` - User guide
- `docs/DEPLOYMENT_GUIDE.md` - Deployment instructions
- `docs/TROUBLESHOOTING.md` - Troubleshooting guide

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📞 Support

- **Documentation**: https://forensmart.readthedocs.io
- **Issues**: https://github.com/yourusername/forensmart/issues
- **Email**: support@forensmart.com

---

## 🙏 Acknowledgments

- Built with Streamlit, FastAPI, and AWS
- AI powered by OpenAI and Anthropic
- Community contributions welcome

---

**Last Updated**: November 25, 2025
**Version**: 1.0.0
**Status**: Production Ready 🚀
