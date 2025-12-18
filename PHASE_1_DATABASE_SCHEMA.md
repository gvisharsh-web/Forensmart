# PHASE 1: DATABASE SCHEMA - DETAILED IMPLEMENTATION

**Date**: December 1, 2025  
**Time**: 17:15 UTC+05:30  
**Duration**: 30 minutes  
**Status**: [READY TO IMPLEMENT]

---

## 🎯 OBJECTIVE

Create database tables for storing approval links, approvals, and history.

---

## 📋 STEP-BY-STEP IMPLEMENTATION

### **Step 1: Create Database Models File**

**File**: `c:\Forensmart\modules\database\consent_approval_schema.py`

**Content**:

```python
"""
Consent Approval Database Schema
Defines SQLAlchemy models for approval links, approvals, and history
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import uuid

Base = declarative_base()


class ApprovalLink(Base):
    """Stores approval links for nominees"""
    __tablename__ = 'approval_links'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    nominee_email = Column(String(255), nullable=False)
    consent_level = Column(String(50), nullable=False)  # STANDARD, LEGAL, FULL
    approval_method = Column(String(50))  # PIN, PATTERN, BIOMETRIC
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='pending')  # pending, approved, expired, revoked
    
    # Relationships
    approvals = relationship('ConsentApproval', back_populates='approval_link')
    history = relationship('ApprovalHistory', back_populates='approval_link')
    
    def is_expired(self):
        """Check if link is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self):
        """Check if link is valid"""
        return self.status == 'pending' and not self.is_expired()
    
    @staticmethod
    def generate_token():
        """Generate unique token"""
        return str(uuid.uuid4())


class ConsentApproval(Base):
    """Stores consent approvals"""
    __tablename__ = 'consent_approvals'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    nominee_email = Column(String(255), nullable=False)
    approval_link_id = Column(Integer, ForeignKey('approval_links.id'))
    consent_level = Column(String(50), nullable=False)  # STANDARD, LEGAL, FULL
    approval_method = Column(String(50))  # PIN, PATTERN, BIOMETRIC
    approved_at = Column(DateTime)
    approved_by = Column(String(255))
    status = Column(String(50), default='pending')  # pending, approved, rejected, revoked
    pin_hash = Column(String(255))  # Hashed PIN for verification
    pattern_hash = Column(String(255))  # Hashed pattern for verification
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    approval_link = relationship('ApprovalLink', back_populates='approvals')
    history = relationship('ApprovalHistory', back_populates='approval')
    
    def approve(self, approval_method, pin_code=None, pattern=None):
        """Mark as approved"""
        self.status = 'approved'
        self.approved_at = datetime.utcnow()
        self.approval_method = approval_method
        if pin_code:
            self.pin_hash = self._hash_value(pin_code)
        if pattern:
            self.pattern_hash = self._hash_value(pattern)
    
    @staticmethod
    def _hash_value(value):
        """Hash a value for storage"""
        import hashlib
        return hashlib.sha256(str(value).encode()).hexdigest()


class ApprovalHistory(Base):
    """Stores approval events for audit trail"""
    __tablename__ = 'approval_history'
    
    id = Column(Integer, primary_key=True)
    case_id = Column(String(255), nullable=False, index=True)
    approval_link_id = Column(Integer, ForeignKey('approval_links.id'))
    approval_id = Column(Integer, ForeignKey('consent_approvals.id'))
    action = Column(String(100), nullable=False)  # link_generated, link_accessed, approved, rejected, revoked
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_email = Column(String(255))
    ip_address = Column(String(50))
    
    # Relationships
    approval_link = relationship('ApprovalLink', back_populates='history')
    approval = relationship('ConsentApproval', back_populates='history')


# Export for use in other modules
__all__ = ['Base', 'ApprovalLink', 'ConsentApproval', 'ApprovalHistory']
```

---

### **Step 2: Create Database Migration Script**

**File**: `c:\Forensmart\modules\database\create_approval_tables.py`

**Content**:

```python
"""
Create approval tables in PostgreSQL database
Run this script to initialize the database schema
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from consent_approval_schema import Base, ApprovalLink, ConsentApproval, ApprovalHistory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from .env
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

print(f"[INFO] Connecting to database: {DATABASE_URL}")

try:
    # Create engine
    engine = create_engine(DATABASE_URL, echo=True)
    
    # Create all tables
    print("[INFO] Creating approval tables...")
    Base.metadata.create_all(engine)
    
    print("[OK] ✅ Approval tables created successfully!")
    print("")
    print("Tables created:")
    print("  ✅ approval_links")
    print("  ✅ consent_approvals")
    print("  ✅ approval_history")
    print("")
    
except Exception as e:
    print(f"[ERROR] ❌ Failed to create tables: {str(e)}")
    raise
```

---

### **Step 3: Run Migration Script**

**Command**:

```bash
cd c:\Forensmart\modules\database
python create_approval_tables.py
```

**Expected Output**:

```
[INFO] Connecting to database: postgresql://forensmart_user:Viszz290@localhost:5432/forensmart
[INFO] Creating approval tables...
[OK] ✅ Approval tables created successfully!

Tables created:
  ✅ approval_links
  ✅ consent_approvals
  ✅ approval_history
```

---

### **Step 4: Verify Tables in PostgreSQL**

**Command**:

```bash
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "\dt"
```

**Expected Output**:

```
                    List of relations
 Schema |        Name        | Type  |      Owner
--------+--------------------+-------+------------------
 public | approval_history   | table | forensmart_user
 public | approval_links     | table | forensmart_user
 public | consent_approvals  | table | forensmart_user
```

---

### **Step 5: Verify Table Structure**

**Command**:

```bash
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U forensmart_user -h localhost -d forensmart -c "\d approval_links"
```

**Expected Output**:

```
                                    Table "public.approval_links"
      Column      |            Type             | Collation | Nullable |      Default
------------------+-----------------------------+-----------+----------+-------------------
 id               | integer                     |           | not null | nextval('approval_links_id_seq'::regclass)
 case_id          | character varying(255)      |           | not null |
 token            | character varying(255)      |           | not null |
 nominee_email    | character varying(255)      |           | not null |
 consent_level    | character varying(50)       |           | not null |
 approval_method  | character varying(50)       |           |          |
 expires_at       | timestamp without time zone |           | not null |
 created_at       | timestamp without time zone |           |          | CURRENT_TIMESTAMP
 status           | character varying(50)       |           |          | 'pending'::character varying
```

---

## ✅ PHASE 1 CHECKLIST

- [ ] Create `modules/database/consent_approval_schema.py`
- [ ] Create `modules/database/create_approval_tables.py`
- [ ] Run migration script
- [ ] Verify tables created in PostgreSQL
- [ ] Verify table structure
- [ ] All 3 tables present:
  - [ ] approval_links
  - [ ] consent_approvals
  - [ ] approval_history

---

## 📊 DATABASE SCHEMA SUMMARY

### **approval_links Table**
```
Stores approval links sent to nominees

Columns:
  - id: Primary key
  - case_id: Reference to case
  - token: Unique approval token
  - nominee_email: Email of nominee
  - consent_level: STANDARD, LEGAL, or FULL
  - approval_method: PIN, PATTERN, or BIOMETRIC
  - expires_at: When link expires
  - created_at: When link was created
  - status: pending, approved, expired, revoked
```

### **consent_approvals Table**
```
Stores actual approvals from nominees

Columns:
  - id: Primary key
  - case_id: Reference to case
  - nominee_email: Email of nominee
  - approval_link_id: Reference to approval link
  - consent_level: STANDARD, LEGAL, or FULL
  - approval_method: PIN, PATTERN, or BIOMETRIC
  - approved_at: When approval happened
  - approved_by: Who approved
  - status: pending, approved, rejected, revoked
  - pin_hash: Hashed PIN (if PIN method used)
  - pattern_hash: Hashed pattern (if pattern method used)
  - created_at: When record created
```

### **approval_history Table**
```
Stores all events for audit trail

Columns:
  - id: Primary key
  - case_id: Reference to case
  - approval_link_id: Reference to approval link
  - approval_id: Reference to approval
  - action: link_generated, link_accessed, approved, rejected, revoked
  - details: Additional details
  - timestamp: When event occurred
  - user_email: Who performed action
  - ip_address: IP address of user
```

---

## 🎯 NEXT PHASE

After Phase 1 is complete:
→ Move to **Phase 2: Database Operations** (30 minutes)

---

## 📝 NOTES

- All tables use PostgreSQL
- All timestamps in UTC
- All tokens are UUIDs
- All sensitive data (PIN, pattern) is hashed
- Audit trail captures all events
- Foreign keys ensure referential integrity

---

**Status**: [READY TO IMPLEMENT]  
**Estimated Time**: 30 minutes  
**Difficulty**: Easy  
**Next Step**: Create files and run migration script

