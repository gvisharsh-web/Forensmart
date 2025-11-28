"""
ANALYSIS MODULE - Unified Database, API, and Models
Comprehensive module for fraud database and analysis API

This module provides:
- Database models (SQLAlchemy ORM)
- Database operations (CRUD)
- FastAPI application
- REST endpoints
- Pydantic validation models
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# API imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:password@localhost:5432/forensmart'
)

engine = create_engine(
    DATABASE_URL,
    echo=os.getenv('SQL_ECHO', 'false').lower() == 'true',
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============================================================================
# SQLALCHEMY MODELS
# ============================================================================

class Fraudster(Base):
    """Fraudster phone numbers and details"""
    __tablename__ = "fraudsters"
    
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String(20), unique=True, index=True, nullable=False)
    fraud_type = Column(String(50), nullable=False)
    name = Column(String(100), nullable=True)
    reports = Column(Integer, default=1)
    last_reported = Column(DateTime, default=datetime.utcnow)
    methods = Column(JSON, nullable=True)
    risk_level = Column(String(20), default='MEDIUM')
    status = Column(String(20), default='ACTIVE')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Harasser(Base):
    """Harasser phone numbers and details"""
    __tablename__ = "harassers"
    
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    reports = Column(Integer, default=1)
    last_reported = Column(DateTime, default=datetime.utcnow)
    harassment_type = Column(String(50), nullable=False)
    victims = Column(Integer, default=1)
    risk_level = Column(String(20), default='MEDIUM')
    status = Column(String(20), default='ACTIVE')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FraudsterEmail(Base):
    """Fraudster email addresses"""
    __tablename__ = "fraudster_emails"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    fraud_type = Column(String(50), nullable=False)
    spoofs = Column(String(100), nullable=True)
    reports = Column(Integer, default=1)
    last_reported = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String(20), default='MEDIUM')
    status = Column(String(20), default='ACTIVE')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SuspiciousLocation(Base):
    """Suspicious locations (fraud hotspots)"""
    __tablename__ = "suspicious_locations"
    
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(String(20), nullable=False)
    longitude = Column(String(20), nullable=False)
    location_name = Column(String(100), nullable=False)
    location_type = Column(String(50), nullable=False)
    reports = Column(Integer, default=1)
    last_reported = Column(DateTime, default=datetime.utcnow)
    known_fraudsters = Column(JSON, nullable=True)
    known_harassers = Column(JSON, nullable=True)
    risk_level = Column(String(20), default='MEDIUM')
    status = Column(String(20), default='ACTIVE')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FraudPattern(Base):
    """Known fraud patterns"""
    __tablename__ = "fraud_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_name = Column(String(100), unique=True, index=True, nullable=False)
    keywords = Column(JSON, nullable=False)
    common_senders = Column(JSON, nullable=True)
    common_numbers = Column(JSON, nullable=True)
    risk_score = Column(Float, default=0.5)
    reports = Column(Integer, default=0)
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisReport(Base):
    """Analysis reports for tracking"""
    __tablename__ = "analysis_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)
    report_type = Column(String(50), nullable=False)
    data = Column(JSON, nullable=False)
    risk_level = Column(String(20), default='MEDIUM')
    created_at = Column(DateTime, default=datetime.utcnow)


class GPSLinkLog(Base):
    """GPS link tracking (WhatsApp, Google Maps, etc.)"""
    __tablename__ = "gps_link_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)
    link = Column(String(500), nullable=False)
    source = Column(String(50), index=True, nullable=False)  # whatsapp, google_maps, geo_url, etc.
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_name = Column(String(200), nullable=True)
    added_by = Column(String(100), nullable=True)
    added_at = Column(DateTime, default=datetime.utcnow, index=True)
    analyzed_at = Column(DateTime, nullable=True)
    risk_level = Column(String(20), default='MEDIUM')
    anomalies_detected = Column(Integer, default=0)
    analysis_data = Column(JSON, nullable=True)
    status = Column(String(20), default='ACTIVE')
    notes = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# PYDANTIC MODELS (Request/Response)
# ============================================================================

class FraudsterCreate(BaseModel):
    """Create fraudster request"""
    phone: str
    fraud_type: str
    name: Optional[str] = None
    methods: Optional[List[str]] = None
    risk_level: str = 'MEDIUM'


class FraudsterResponse(BaseModel):
    """Fraudster response"""
    id: int
    phone: str
    fraud_type: str
    name: Optional[str]
    reports: int
    risk_level: str
    status: str
    last_reported: datetime
    
    class Config:
        from_attributes = True


class HarasserCreate(BaseModel):
    """Create harasser request"""
    phone: str
    harassment_type: str
    name: Optional[str] = None
    risk_level: str = 'MEDIUM'


class HarasserResponse(BaseModel):
    """Harasser response"""
    id: int
    phone: str
    harassment_type: str
    name: Optional[str]
    reports: int
    risk_level: str
    status: str
    last_reported: datetime
    
    class Config:
        from_attributes = True


class LocationCreate(BaseModel):
    """Create location request"""
    latitude: str
    longitude: str
    location_name: str
    location_type: str
    risk_level: str = 'MEDIUM'


class LocationResponse(BaseModel):
    """Location response"""
    id: int
    latitude: str
    longitude: str
    location_name: str
    location_type: str
    reports: int
    risk_level: str
    status: str
    known_fraudsters: Optional[List[str]]
    known_harassers: Optional[List[str]]
    
    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Statistics response"""
    total_fraudsters: int
    total_harassers: int
    total_emails: int
    total_locations: int
    total_patterns: int
    critical_fraudsters: int
    critical_harassers: int


class GPSLinkCreate(BaseModel):
    """Create GPS link log request"""
    case_id: str
    link: str
    source: str  # whatsapp, google_maps, geo_url, etc.
    latitude: float
    longitude: float
    location_name: Optional[str] = None
    added_by: Optional[str] = None
    notes: Optional[str] = None


class GPSLinkResponse(BaseModel):
    """GPS link log response"""
    id: int
    case_id: str
    link: str
    source: str
    latitude: float
    longitude: float
    location_name: Optional[str]
    added_by: Optional[str]
    added_at: datetime
    analyzed_at: Optional[datetime]
    risk_level: str
    anomalies_detected: int
    status: str
    
    class Config:
        from_attributes = True


class GPSLinkAnalysis(BaseModel):
    """GPS link analysis request"""
    link_id: int
    risk_level: str
    anomalies_detected: int
    analysis_data: Optional[Dict[str, Any]] = None


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manage database operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    # Fraudster operations
    def add_fraudster(self, phone: str, fraud_type: str, name: str = None, 
                     methods: List[str] = None, risk_level: str = 'MEDIUM') -> Fraudster:
        """Add new fraudster"""
        fraudster = Fraudster(
            phone=phone,
            fraud_type=fraud_type,
            name=name,
            methods=methods,
            risk_level=risk_level
        )
        self.db.add(fraudster)
        self.db.commit()
        self.db.refresh(fraudster)
        logger.info(f"✅ Added fraudster: {phone}")
        return fraudster
    
    def get_fraudster(self, phone: str) -> Optional[Fraudster]:
        """Get fraudster by phone"""
        return self.db.query(Fraudster).filter(Fraudster.phone == phone).first()
    
    def report_fraudster(self, phone: str) -> Optional[Fraudster]:
        """Report fraudster (increment reports)"""
        fraudster = self.get_fraudster(phone)
        if fraudster:
            fraudster.reports += 1
            fraudster.last_reported = datetime.utcnow()
            self.db.commit()
            self.db.refresh(fraudster)
            logger.info(f"✅ Reported fraudster: {phone} (reports: {fraudster.reports})")
        return fraudster
    
    def get_all_fraudsters(self, risk_level: str = None) -> List[Fraudster]:
        """Get all fraudsters, optionally filtered by risk level"""
        query = self.db.query(Fraudster)
        if risk_level:
            query = query.filter(Fraudster.risk_level == risk_level)
        return query.all()
    
    # Harasser operations
    def add_harasser(self, phone: str, harassment_type: str, name: str = None,
                    risk_level: str = 'MEDIUM') -> Harasser:
        """Add new harasser"""
        harasser = Harasser(
            phone=phone,
            harassment_type=harassment_type,
            name=name,
            risk_level=risk_level
        )
        self.db.add(harasser)
        self.db.commit()
        self.db.refresh(harasser)
        logger.info(f"✅ Added harasser: {phone}")
        return harasser
    
    def get_harasser(self, phone: str) -> Optional[Harasser]:
        """Get harasser by phone"""
        return self.db.query(Harasser).filter(Harasser.phone == phone).first()
    
    def report_harasser(self, phone: str) -> Optional[Harasser]:
        """Report harasser (increment reports)"""
        harasser = self.get_harasser(phone)
        if harasser:
            harasser.reports += 1
            harasser.last_reported = datetime.utcnow()
            self.db.commit()
            self.db.refresh(harasser)
            logger.info(f"✅ Reported harasser: {phone} (reports: {harasser.reports})")
        return harasser
    
    # Location operations
    def add_location(self, latitude: str, longitude: str, location_name: str,
                    location_type: str, risk_level: str = 'MEDIUM') -> SuspiciousLocation:
        """Add suspicious location"""
        location = SuspiciousLocation(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            location_type=location_type,
            risk_level=risk_level
        )
        self.db.add(location)
        self.db.commit()
        self.db.refresh(location)
        logger.info(f"✅ Added location: {location_name}")
        return location
    
    def get_location(self, latitude: str, longitude: str) -> Optional[SuspiciousLocation]:
        """Get location by coordinates"""
        return self.db.query(SuspiciousLocation).filter(
            SuspiciousLocation.latitude == latitude,
            SuspiciousLocation.longitude == longitude
        ).first()
    
    # GPS Link Tracking
    def add_gps_link(self, case_id: str, link: str, source: str, 
                    latitude: float, longitude: float, location_name: str = None,
                    added_by: str = None, notes: str = None) -> GPSLinkLog:
        """Add GPS link to tracking database"""
        gps_link = GPSLinkLog(
            case_id=case_id,
            link=link,
            source=source,
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            added_by=added_by,
            notes=notes
        )
        self.db.add(gps_link)
        self.db.commit()
        self.db.refresh(gps_link)
        logger.info(f"✅ GPS link tracked: {source} ({latitude}, {longitude})")
        return gps_link
    
    def get_gps_link(self, link_id: int) -> Optional[GPSLinkLog]:
        """Get GPS link by ID"""
        return self.db.query(GPSLinkLog).filter(GPSLinkLog.id == link_id).first()
    
    def get_gps_links_by_case(self, case_id: str) -> List[GPSLinkLog]:
        """Get all GPS links for a case"""
        return self.db.query(GPSLinkLog).filter(GPSLinkLog.case_id == case_id).all()
    
    def get_gps_links_by_source(self, source: str) -> List[GPSLinkLog]:
        """Get all GPS links by source (whatsapp, google_maps, etc.)"""
        return self.db.query(GPSLinkLog).filter(GPSLinkLog.source == source).all()
    
    def update_gps_link_analysis(self, link_id: int, risk_level: str, 
                                anomalies_detected: int, analysis_data: Dict[str, Any] = None) -> Optional[GPSLinkLog]:
        """Update GPS link with analysis results"""
        gps_link = self.get_gps_link(link_id)
        if gps_link:
            gps_link.risk_level = risk_level
            gps_link.anomalies_detected = anomalies_detected
            gps_link.analysis_data = analysis_data
            gps_link.analyzed_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(gps_link)
            logger.info(f"✅ GPS link analysis updated: {link_id}")
        return gps_link
    
    def get_gps_links_statistics(self, case_id: str = None) -> Dict[str, Any]:
        """Get GPS links statistics"""
        query = self.db.query(GPSLinkLog)
        if case_id:
            query = query.filter(GPSLinkLog.case_id == case_id)
        
        total = query.count()
        by_source = {}
        for source in ['whatsapp', 'google_maps', 'geo_url']:
            count = query.filter(GPSLinkLog.source == source).count()
            if count > 0:
                by_source[source] = count
        
        high_risk = query.filter(GPSLinkLog.risk_level == 'HIGH').count()
        critical = query.filter(GPSLinkLog.risk_level == 'CRITICAL').count()
        
        return {
            'total_links': total,
            'by_source': by_source,
            'high_risk': high_risk,
            'critical': critical
        }
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_fraudsters': self.db.query(Fraudster).count(),
            'total_harassers': self.db.query(Harasser).count(),
            'total_emails': self.db.query(FraudsterEmail).count(),
            'total_locations': self.db.query(SuspiciousLocation).count(),
            'total_patterns': self.db.query(FraudPattern).count(),
            'critical_fraudsters': self.db.query(Fraudster).filter(
                Fraudster.risk_level == 'CRITICAL'
            ).count(),
            'critical_harassers': self.db.query(Harasser).filter(
                Harasser.risk_level == 'CRITICAL'
            ).count()
        }
    
    # Dashboard Methods
    def get_dashboard_summary(self, case_id: str = None, dev_mode: bool = False) -> Dict[str, Any]:
        """Get dashboard summary statistics with dev mode support"""
        try:
            stats = self.get_statistics()
            gps_stats = self.get_gps_links_statistics(case_id) if case_id else {}
            
            summary = {
                'total_fraudsters': stats['total_fraudsters'],
                'total_harassers': stats['total_harassers'],
                'total_gps_links': gps_stats.get('total_links', 0),
                'critical_cases': stats['critical_fraudsters'] + stats['critical_harassers'],
                'high_risk_locations': gps_stats.get('high_risk', 0),
                'critical_fraudsters': stats['critical_fraudsters'],
                'critical_harassers': stats['critical_harassers'],
                'dev_mode': dev_mode
            }
            
            if dev_mode:
                logger.info("🧪 Dashboard summary retrieved in dev mode")
            
            return summary
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {'error': str(e), 'dev_mode': dev_mode}
    
    def get_fraudster_risk_distribution(self) -> Dict[str, int]:
        """Get fraudster risk level distribution"""
        fraudsters = self.db.query(Fraudster).all()
        distribution = {}
        
        for fraudster in fraudsters:
            risk = fraudster.risk_level
            distribution[risk] = distribution.get(risk, 0) + 1
        
        return distribution
    
    def get_harasser_risk_distribution(self) -> Dict[str, int]:
        """Get harasser risk level distribution"""
        harassers = self.db.query(Harasser).all()
        distribution = {}
        
        for harasser in harassers:
            risk = harasser.risk_level
            distribution[risk] = distribution.get(risk, 0) + 1
        
        return distribution
    
    def get_recent_gps_links(self, case_id: str, limit: int = 5) -> List[GPSLinkLog]:
        """Get recent GPS links for a case"""
        return self.db.query(GPSLinkLog).filter(
            GPSLinkLog.case_id == case_id
        ).order_by(GPSLinkLog.added_at.desc()).limit(limit).all()
    
    def get_case_summary(self, case_id: str, dev_mode: bool = False) -> Dict[str, Any]:
        """Get case summary information with dev mode support"""
        try:
            gps_stats = self.get_gps_links_statistics(case_id)
            stats = self.get_statistics()
            
            summary = {
                'case_id': case_id,
                'total_fraudsters': stats['total_fraudsters'],
                'total_harassers': stats['total_harassers'],
                'total_gps_links': gps_stats.get('total_links', 0),
                'whatsapp_links': gps_stats.get('by_source', {}).get('whatsapp', 0),
                'google_maps_links': gps_stats.get('by_source', {}).get('google_maps', 0),
                'high_risk_items': stats['critical_fraudsters'] + stats['critical_harassers'] + gps_stats.get('high_risk', 0),
                'dev_mode': dev_mode
            }
            
            if dev_mode:
                logger.info(f"🧪 Case summary retrieved in dev mode for {case_id}")
            
            return summary
        except Exception as e:
            logger.error(f"Error getting case summary: {e}")
            return {'error': str(e), 'dev_mode': dev_mode}
    
    def get_risk_summary(self, case_id: str = None, dev_mode: bool = False) -> Dict[str, Any]:
        """Get risk summary for dashboard with dev mode support"""
        try:
            stats = self.get_statistics()
            gps_stats = self.get_gps_links_statistics(case_id) if case_id else {}
            
            summary = {
                'critical_fraudsters': stats['critical_fraudsters'],
                'critical_harassers': stats['critical_harassers'],
                'high_risk_gps_locations': gps_stats.get('high_risk', 0),
                'total_critical': stats['critical_fraudsters'] + stats['critical_harassers'] + gps_stats.get('high_risk', 0),
                'dev_mode': dev_mode
            }
            
            if dev_mode:
                logger.info("🧪 Risk summary retrieved in dev mode")
            
            return summary
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {'error': str(e), 'dev_mode': dev_mode}


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database - create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Forensmart Analysis API",
    description="API for fraud and harassment database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting Forensmart Analysis API...")
    init_database()
    logger.info("✅ API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Forensmart Analysis API...")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Forensmart Analysis API",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# FRAUDSTER ENDPOINTS
# ============================================================================

@app.get("/fraudsters/{phone}", response_model=FraudsterResponse)
async def get_fraudster(phone: str):
    """Get fraudster by phone number"""
    with DatabaseManager() as db:
        fraudster = db.get_fraudster(phone)
        if not fraudster:
            raise HTTPException(status_code=404, detail="Fraudster not found")
        return fraudster


@app.post("/fraudsters/", response_model=FraudsterResponse)
async def create_fraudster(fraudster_data: FraudsterCreate):
    """Create new fraudster"""
    with DatabaseManager() as db:
        existing = db.get_fraudster(fraudster_data.phone)
        if existing:
            raise HTTPException(status_code=400, detail="Fraudster already exists")
        
        fraudster = db.add_fraudster(
            phone=fraudster_data.phone,
            fraud_type=fraudster_data.fraud_type,
            name=fraudster_data.name,
            methods=fraudster_data.methods,
            risk_level=fraudster_data.risk_level
        )
        return fraudster


@app.put("/fraudsters/{phone}/report", response_model=FraudsterResponse)
async def report_fraudster(phone: str):
    """Report fraudster (increment reports)"""
    with DatabaseManager() as db:
        fraudster = db.report_fraudster(phone)
        if not fraudster:
            raise HTTPException(status_code=404, detail="Fraudster not found")
        return fraudster


@app.get("/fraudsters/", response_model=List[FraudsterResponse])
async def list_fraudsters(risk_level: Optional[str] = Query(None)):
    """List all fraudsters, optionally filtered by risk level"""
    with DatabaseManager() as db:
        fraudsters = db.get_all_fraudsters(risk_level=risk_level)
        return fraudsters


# ============================================================================
# HARASSER ENDPOINTS
# ============================================================================

@app.get("/harassers/{phone}", response_model=HarasserResponse)
async def get_harasser(phone: str):
    """Get harasser by phone number"""
    with DatabaseManager() as db:
        harasser = db.get_harasser(phone)
        if not harasser:
            raise HTTPException(status_code=404, detail="Harasser not found")
        return harasser


@app.post("/harassers/", response_model=HarasserResponse)
async def create_harasser(harasser_data: HarasserCreate):
    """Create new harasser"""
    with DatabaseManager() as db:
        existing = db.get_harasser(harasser_data.phone)
        if existing:
            raise HTTPException(status_code=400, detail="Harasser already exists")
        
        harasser = db.add_harasser(
            phone=harasser_data.phone,
            harassment_type=harasser_data.harassment_type,
            name=harasser_data.name,
            risk_level=harasser_data.risk_level
        )
        return harasser


@app.put("/harassers/{phone}/report", response_model=HarasserResponse)
async def report_harasser(phone: str):
    """Report harasser (increment reports)"""
    with DatabaseManager() as db:
        harasser = db.report_harasser(phone)
        if not harasser:
            raise HTTPException(status_code=404, detail="Harasser not found")
        return harasser


# ============================================================================
# LOCATION ENDPOINTS
# ============================================================================

@app.get("/locations/{latitude},{longitude}", response_model=LocationResponse)
async def get_location(latitude: str, longitude: str):
    """Get suspicious location by coordinates"""
    with DatabaseManager() as db:
        location = db.get_location(latitude, longitude)
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        return location


@app.post("/locations/", response_model=LocationResponse)
async def create_location(location_data: LocationCreate):
    """Create new suspicious location"""
    with DatabaseManager() as db:
        existing = db.get_location(location_data.latitude, location_data.longitude)
        if existing:
            raise HTTPException(status_code=400, detail="Location already exists")
        
        location = db.add_location(
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            location_name=location_data.location_name,
            location_type=location_data.location_type,
            risk_level=location_data.risk_level
        )
        return location


# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.get("/statistics/", response_model=StatisticsResponse)
async def get_statistics():
    """Get database statistics"""
    with DatabaseManager() as db:
        stats = db.get_statistics()
        return stats


@app.get("/statistics/top-fraudsters")
async def get_top_fraudsters(limit: int = Query(10, ge=1, le=100)):
    """Get top fraudsters by reports"""
    with DatabaseManager() as db:
        fraudsters = db.db.query(Fraudster).order_by(
            Fraudster.reports.desc()
        ).limit(limit).all()
        return [
            {
                "phone": f.phone,
                "fraud_type": f.fraud_type,
                "reports": f.reports,
                "risk_level": f.risk_level
            }
            for f in fraudsters
        ]


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@app.get("/search/phone/{phone}")
async def search_phone(phone: str):
    """Search for phone in both fraudsters and harassers"""
    with DatabaseManager() as db:
        fraudster = db.get_fraudster(phone)
        harasser = db.get_harasser(phone)
        
        return {
            "phone": phone,
            "fraudster": fraudster.__dict__ if fraudster else None,
            "harasser": harasser.__dict__ if harasser else None,
            "found": fraudster is not None or harasser is not None
        }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Forensmart Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "fraudsters": "/fraudsters/",
            "harassers": "/harassers/",
            "locations": "/locations/",
            "statistics": "/statistics/",
            "search": "/search/phone/{phone}"
        }
    }


# ============================================================================
# AUTO-UPDATE SYSTEM
# ============================================================================

class DatabaseAutoUpdater:
    """Automatic database update system"""
    
    def __init__(self):
        self.last_update = None
        self.update_interval = 3600  # 1 hour in seconds
    
    def should_update(self) -> bool:
        """Check if database should be updated"""
        if self.last_update is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_update).total_seconds()
        return elapsed >= self.update_interval
    
    def auto_report_fraudster(self, phone: str) -> bool:
        """Auto-report fraudster when detected"""
        try:
            with DatabaseManager() as db:
                fraudster = db.report_fraudster(phone)
                if fraudster:
                    logger.info(f"✅ Auto-reported fraudster: {phone} (reports: {fraudster.reports})")
                    return True
        except Exception as e:
            logger.error(f"❌ Auto-report failed: {e}")
        return False
    
    def auto_report_harasser(self, phone: str) -> bool:
        """Auto-report harasser when detected"""
        try:
            with DatabaseManager() as db:
                harasser = db.report_harasser(phone)
                if harasser:
                    logger.info(f"✅ Auto-reported harasser: {phone} (reports: {harasser.reports})")
                    return True
        except Exception as e:
            logger.error(f"❌ Auto-report failed: {e}")
        return False
    
    def bulk_add_fraudsters(self, fraudsters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk add fraudsters to database"""
        results = {"added": 0, "failed": 0, "skipped": 0}
        
        with DatabaseManager() as db:
            for fraudster_data in fraudsters:
                try:
                    existing = db.get_fraudster(fraudster_data["phone"])
                    if existing:
                        results["skipped"] += 1
                        logger.info(f"⏭️ Fraudster already exists: {fraudster_data['phone']}")
                        continue
                    
                    db.add_fraudster(
                        phone=fraudster_data["phone"],
                        fraud_type=fraudster_data.get("fraud_type", "UNKNOWN"),
                        name=fraudster_data.get("name"),
                        methods=fraudster_data.get("methods"),
                        risk_level=fraudster_data.get("risk_level", "MEDIUM")
                    )
                    results["added"] += 1
                    logger.info(f"✅ Added fraudster: {fraudster_data['phone']}")
                except Exception as e:
                    results["failed"] += 1
                    logger.error(f"❌ Failed to add fraudster: {e}")
        
        return results
    
    def bulk_add_harassers(self, harassers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk add harassers to database"""
        results = {"added": 0, "failed": 0, "skipped": 0}
        
        with DatabaseManager() as db:
            for harasser_data in harassers:
                try:
                    existing = db.get_harasser(harasser_data["phone"])
                    if existing:
                        results["skipped"] += 1
                        logger.info(f"⏭️ Harasser already exists: {harasser_data['phone']}")
                        continue
                    
                    db.add_harasser(
                        phone=harasser_data["phone"],
                        harassment_type=harasser_data.get("harassment_type", "UNKNOWN"),
                        name=harasser_data.get("name"),
                        risk_level=harasser_data.get("risk_level", "MEDIUM")
                    )
                    results["added"] += 1
                    logger.info(f"✅ Added harasser: {harasser_data['phone']}")
                except Exception as e:
                    results["failed"] += 1
                    logger.error(f"❌ Failed to add harasser: {e}")
        
        return results
    
    def bulk_add_locations(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk add suspicious locations to database"""
        results = {"added": 0, "failed": 0, "skipped": 0}
        
        with DatabaseManager() as db:
            for location_data in locations:
                try:
                    existing = db.get_location(location_data["latitude"], location_data["longitude"])
                    if existing:
                        results["skipped"] += 1
                        logger.info(f"⏭️ Location already exists: {location_data['location_name']}")
                        continue
                    
                    db.add_location(
                        latitude=location_data["latitude"],
                        longitude=location_data["longitude"],
                        location_name=location_data.get("location_name", "Unknown"),
                        location_type=location_data.get("location_type", "UNKNOWN"),
                        risk_level=location_data.get("risk_level", "MEDIUM")
                    )
                    results["added"] += 1
                    logger.info(f"✅ Added location: {location_data['location_name']}")
                except Exception as e:
                    results["failed"] += 1
                    logger.error(f"❌ Failed to add location: {e}")
        
        return results


# ============================================================================
# AUTO-UPDATE API ENDPOINTS
# ============================================================================

updater = DatabaseAutoUpdater()

@app.post("/admin/update/fraudsters")
async def update_fraudsters(fraudsters: List[Dict[str, Any]]):
    """Admin endpoint: Bulk update fraudsters"""
    results = updater.bulk_add_fraudsters(fraudsters)
    return {
        "status": "updated",
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/admin/update/harassers")
async def update_harassers(harassers: List[Dict[str, Any]]):
    """Admin endpoint: Bulk update harassers"""
    results = updater.bulk_add_harassers(harassers)
    return {
        "status": "updated",
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/admin/update/locations")
async def update_locations(locations: List[Dict[str, Any]]):
    """Admin endpoint: Bulk update locations"""
    results = updater.bulk_add_locations(locations)
    return {
        "status": "updated",
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/admin/report/fraudster/{phone}")
async def auto_report_fraudster(phone: str):
    """Auto-report fraudster when detected"""
    success = updater.auto_report_fraudster(phone)
    return {
        "status": "reported" if success else "failed",
        "phone": phone,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/admin/report/harasser/{phone}")
async def auto_report_harasser(phone: str):
    """Auto-report harasser when detected"""
    success = updater.auto_report_harasser(phone)
    return {
        "status": "reported" if success else "failed",
        "phone": phone,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
