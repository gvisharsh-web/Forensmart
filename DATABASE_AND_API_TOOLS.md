# 🗄️ DATABASE & API TOOLS - COMPREHENSIVE GUIDE

**Status**: Full Database & API Stack Available
**Date**: November 25, 2025

---

## ✅ YES - WE HAVE COMPLETE DATABASE & API TOOLS!

---

## 📊 DATABASE TOOLS (Lines 33-36)

### 1. SQLAlchemy ✅

**What it is:**
- Python SQL toolkit and Object-Relational Mapping (ORM)
- Works with multiple databases
- Version: 2.0.0+

**In requirements.txt (line 34):**
```
sqlalchemy>=2.0.0
```

**Supported Databases:**
- PostgreSQL
- MySQL
- SQLite
- Oracle
- Microsoft SQL Server

**Features:**
- ORM (Object-Relational Mapping)
- Query builder
- Connection pooling
- Transaction management
- Schema creation

**Example Usage:**
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database connection
engine = create_engine('postgresql://user:password@localhost/forensmart')

# Define model
Base = declarative_base()

class Fraudster(Base):
    __tablename__ = 'fraudsters'
    
    id = Column(Integer, primary_key=True)
    phone = Column(String(20), unique=True)
    fraud_type = Column(String(50))
    reports = Column(Integer)

# Create tables
Base.metadata.create_all(engine)

# Session for queries
Session = sessionmaker(bind=engine)
session = Session()

# Add fraudster
fraudster = Fraudster(phone="+1-555-0100", fraud_type="PHISHING", reports=45)
session.add(fraudster)
session.commit()

# Query
result = session.query(Fraudster).filter_by(phone="+1-555-0100").first()
```

---

### 2. PostgreSQL (psycopg2) ✅

**What it is:**
- PostgreSQL database adapter for Python
- Most popular relational database
- Version: 2.9.0+

**In requirements.txt (line 35):**
```
psycopg2-binary>=2.9.0
```

**Features:**
- Full PostgreSQL support
- Connection pooling
- Transactions
- JSON support
- Array support
- Full-text search

**Example Usage:**
```python
import psycopg2

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    database="forensmart",
    user="postgres",
    password="password"
)

cursor = conn.cursor()

# Create table
cursor.execute("""
    CREATE TABLE fraudsters (
        id SERIAL PRIMARY KEY,
        phone VARCHAR(20) UNIQUE,
        fraud_type VARCHAR(50),
        reports INT
    )
""")

# Insert data
cursor.execute(
    "INSERT INTO fraudsters (phone, fraud_type, reports) VALUES (%s, %s, %s)",
    ("+1-555-0100", "PHISHING", 45)
)

conn.commit()
cursor.close()
conn.close()
```

---

### 3. Redis ✅

**What it is:**
- In-memory data store
- Cache layer
- Session storage
- Real-time data

**In requirements.txt (line 36):**
```
redis>=5.0.0
```

**Features:**
- Key-value storage
- Caching
- Session management
- Pub/Sub messaging
- Real-time analytics

**Example Usage:**
```python
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Store fraudster in cache
r.set('fraudster:+1-555-0100', 'PHISHING', ex=3600)

# Retrieve from cache
result = r.get('fraudster:+1-555-0100')

# Store as JSON
import json
fraudster_data = {
    "phone": "+1-555-0100",
    "type": "PHISHING",
    "reports": 45
}
r.set('fraudster:data:+1-555-0100', json.dumps(fraudster_data))

# Retrieve JSON
data = json.loads(r.get('fraudster:data:+1-555-0100'))
```

---

## 🌐 API TOOLS (Lines 45-49)

### 1. FastAPI ✅

**What it is:**
- Modern Python web framework for building APIs
- Fast, easy to use
- Version: 0.104.0+

**In requirements.txt (line 48):**
```
fastapi>=0.104.0
```

**Features:**
- REST API creation
- Automatic documentation (Swagger)
- Data validation (Pydantic)
- Async support
- Built-in security

**Example Usage:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define data model
class Fraudster(BaseModel):
    phone: str
    fraud_type: str
    reports: int

# Create endpoint
@app.get("/fraudsters/{phone}")
async def get_fraudster(phone: str):
    # Query database
    return {"phone": phone, "fraud_type": "PHISHING", "reports": 45}

# Create endpoint with POST
@app.post("/fraudsters/")
async def create_fraudster(fraudster: Fraudster):
    # Save to database
    return {"status": "created", "fraudster": fraudster}

# Run: uvicorn main:app --reload
```

---

### 2. Uvicorn ✅

**What it is:**
- ASGI web server
- Runs FastAPI applications
- Version: 0.24.0+

**In requirements.txt (line 49):**
```
uvicorn>=0.24.0
```

**Features:**
- High performance
- Async support
- WebSocket support
- SSL/TLS support

**Usage:**
```bash
# Run FastAPI app
uvicorn main:app --reload

# Run on specific port
uvicorn main:app --port 8000

# Run with multiple workers
uvicorn main:app --workers 4
```

---

### 3. Requests ✅

**What it is:**
- HTTP library for Python
- Make API calls
- Version: 2.31.0+

**In requirements.txt (line 46):**
```
requests>=2.31.0
```

**Features:**
- GET, POST, PUT, DELETE requests
- JSON handling
- Authentication
- Session management

**Example Usage:**
```python
import requests

# GET request
response = requests.get('http://api.example.com/fraudsters/+1-555-0100')
data = response.json()

# POST request
payload = {
    "phone": "+1-555-0100",
    "fraud_type": "PHISHING",
    "reports": 45
}
response = requests.post('http://api.example.com/fraudsters/', json=payload)

# With headers
headers = {'Authorization': 'Bearer token123'}
response = requests.get('http://api.example.com/fraudsters/', headers=headers)
```

---

### 4. HTTPX ✅

**What it is:**
- Modern HTTP client
- Async support
- Version: 0.25.0+

**In requirements.txt (line 47):**
```
httpx>=0.25.0
```

**Features:**
- Async/await support
- HTTP/2 support
- Streaming
- Better than requests for async

**Example Usage:**
```python
import httpx
import asyncio

async def fetch_fraudster():
    async with httpx.AsyncClient() as client:
        response = await client.get('http://api.example.com/fraudsters/+1-555-0100')
        return response.json()

# Run async
asyncio.run(fetch_fraudster())
```

---

## 🏗️ ARCHITECTURE FOR FRAUD DATABASE

### Option 1: PostgreSQL + SQLAlchemy + FastAPI

```
┌─────────────────────────────────────────────┐
│ Streamlit UI (Frontend)                     │
├─────────────────────────────────────────────┤
│ Makes HTTP requests to API                  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ FastAPI (Backend API)                       │
│ - GET /fraudsters/{phone}                   │
│ - POST /fraudsters/                         │
│ - GET /locations/{lat},{lon}                │
│ - POST /report/                             │
├─────────────────────────────────────────────┤
│ Runs on http://localhost:8000               │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ SQLAlchemy ORM                              │
│ - Fraudster model                           │
│ - Harasser model                            │
│ - Location model                            │
├─────────────────────────────────────────────┤
│ Database abstraction layer                  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ PostgreSQL Database                         │
│ - fraudsters table                          │
│ - harassers table                           │
│ - locations table                           │
│ - fraud_patterns table                      │
└─────────────────────────────────────────────┘
```

---

### Option 2: PostgreSQL + SQLAlchemy + Redis Cache

```
┌─────────────────────────────────────────────┐
│ Streamlit UI (Frontend)                     │
├─────────────────────────────────────────────┤
│ Makes requests                              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ FastAPI (Backend API)                       │
├─────────────────────────────────────────────┤
│ 1. Check Redis cache                        │
│ 2. If not found, query PostgreSQL           │
│ 3. Store in Redis for future requests       │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ Redis   │      │PostgreSQL│
   │ Cache   │      │Database  │
   └─────────┘      └──────────┘
```

---

## 🚀 IMPLEMENTATION PLAN FOR FRAUD DATABASE

### Step 1: Create Database Models

```python
# models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Fraudster(Base):
    __tablename__ = 'fraudsters'
    
    id = Column(Integer, primary_key=True)
    phone = Column(String(20), unique=True)
    fraud_type = Column(String(50))
    name = Column(String(100))
    reports = Column(Integer, default=1)
    last_reported = Column(DateTime, default=datetime.now)
    methods = Column(JSON)
    risk_level = Column(String(20))
    status = Column(String(20), default='ACTIVE')

class Harasser(Base):
    __tablename__ = 'harassers'
    
    id = Column(Integer, primary_key=True)
    phone = Column(String(20), unique=True)
    name = Column(String(100))
    reports = Column(Integer, default=1)
    harassment_type = Column(String(50))
    risk_level = Column(String(20))
    status = Column(String(20), default='ACTIVE')

class SuspiciousLocation(Base):
    __tablename__ = 'suspicious_locations'
    
    id = Column(Integer, primary_key=True)
    latitude = Column(String(20))
    longitude = Column(String(20))
    location_name = Column(String(100))
    location_type = Column(String(50))
    reports = Column(Integer, default=1)
    risk_level = Column(String(20))
    status = Column(String(20), default='ACTIVE')
```

---

### Step 2: Create FastAPI Endpoints

```python
# api.py
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from models import Fraudster, Harasser, SuspiciousLocation

app = FastAPI()

# Database connection
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost/forensmart')
SessionLocal = sessionmaker(bind=engine)

# Fraudster endpoints
@app.get("/fraudsters/{phone}")
async def get_fraudster(phone: str):
    db = SessionLocal()
    fraudster = db.query(Fraudster).filter_by(phone=phone).first()
    if not fraudster:
        raise HTTPException(status_code=404, detail="Fraudster not found")
    return fraudster

@app.post("/fraudsters/")
async def create_fraudster(fraudster_data: dict):
    db = SessionLocal()
    fraudster = Fraudster(**fraudster_data)
    db.add(fraudster)
    db.commit()
    return {"status": "created", "fraudster": fraudster}

@app.put("/fraudsters/{phone}/report")
async def report_fraudster(phone: str):
    db = SessionLocal()
    fraudster = db.query(Fraudster).filter_by(phone=phone).first()
    if fraudster:
        fraudster.reports += 1
        fraudster.last_reported = datetime.now()
        db.commit()
    return {"status": "reported", "reports": fraudster.reports}

# Location endpoints
@app.get("/locations/{lat},{lon}")
async def get_location(lat: str, lon: str):
    db = SessionLocal()
    location = db.query(SuspiciousLocation).filter_by(
        latitude=lat, longitude=lon
    ).first()
    if not location:
        return {"match": False}
    return {"match": True, "location": location}

# Statistics endpoint
@app.get("/statistics/")
async def get_statistics():
    db = SessionLocal()
    return {
        "total_fraudsters": db.query(Fraudster).count(),
        "total_harassers": db.query(Harasser).count(),
        "total_locations": db.query(SuspiciousLocation).count()
    }
```

---

### Step 3: Integrate with Suspicious Classifier

```python
# suspicious_classifier.py
import requests

class SuspiciousClassifier:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def check_phone(self, phone):
        response = requests.get(f"{self.api_url}/fraudsters/{phone}")
        if response.status_code == 200:
            return response.json()
        return {"match": False}
    
    def classify_message(self, message, phone, email):
        # AI classification
        ai_result = self.ai_classify(message)
        
        # Database check
        db_result = self.check_phone(phone)
        
        # Combine results
        combined_risk = (ai_result['confidence'] * 0.6 + 
                        (0.95 if db_result['match'] else 0) * 0.4)
        
        return {
            "ai_result": ai_result,
            "database_match": db_result,
            "combined_risk": combined_risk
        }
```

---

### Step 4: Integrate with Location Intelligence

```python
# location_intelligence.py
import requests

class LocationIntelligence:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def check_location(self, latitude, longitude):
        response = requests.get(
            f"{self.api_url}/locations/{latitude},{longitude}"
        )
        return response.json()
    
    def analyze_location(self, latitude, longitude):
        location_check = self.check_location(latitude, longitude)
        
        if location_check['match']:
            return {
                "location_match": True,
                "location_data": location_check['location'],
                "risk_level": location_check['location']['risk_level']
            }
        
        return {"location_match": False}
```

---

## 📊 DATABASE SCHEMA

```sql
-- Fraudsters table
CREATE TABLE fraudsters (
    id SERIAL PRIMARY KEY,
    phone VARCHAR(20) UNIQUE NOT NULL,
    fraud_type VARCHAR(50),
    name VARCHAR(100),
    reports INT DEFAULT 1,
    last_reported TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    methods JSONB,
    risk_level VARCHAR(20),
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

-- Harassers table
CREATE TABLE harassers (
    id SERIAL PRIMARY KEY,
    phone VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100),
    reports INT DEFAULT 1,
    harassment_type VARCHAR(50),
    risk_level VARCHAR(20),
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

-- Suspicious locations table
CREATE TABLE suspicious_locations (
    id SERIAL PRIMARY KEY,
    latitude VARCHAR(20),
    longitude VARCHAR(20),
    location_name VARCHAR(100),
    location_type VARCHAR(50),
    reports INT DEFAULT 1,
    risk_level VARCHAR(20),
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

-- Fraud patterns table
CREATE TABLE fraud_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100),
    keywords JSONB,
    common_senders JSONB,
    common_numbers JSONB,
    risk_score DECIMAL(3, 2),
    reports INT
);

-- Create indexes for fast queries
CREATE INDEX idx_fraudsters_phone ON fraudsters(phone);
CREATE INDEX idx_harassers_phone ON harassers(phone);
CREATE INDEX idx_locations_coords ON suspicious_locations(latitude, longitude);
```

---

## ✅ COMPLETE DATABASE & API STACK

**Available Tools:**
- ✅ SQLAlchemy (ORM)
- ✅ PostgreSQL (Database)
- ✅ Redis (Cache)
- ✅ FastAPI (API Framework)
- ✅ Uvicorn (Server)
- ✅ Requests (HTTP Client)
- ✅ HTTPX (Async HTTP)

**Ready to Build:**
- ✅ Fraud database
- ✅ REST API endpoints
- ✅ Caching layer
- ✅ Integration with modules

---

## 🚀 READY TO IMPLEMENT PHASE 3 WITH FULL DATABASE

All tools available and ready to use!
