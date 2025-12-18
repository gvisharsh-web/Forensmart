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

print("")
print("=" * 80)
print("CREATING APPROVAL TABLES IN POSTGRESQL")
print("=" * 80)
print("")
print(f"[INFO] Connecting to database...")
print(f"[INFO] Database: forensmart")
print("")

try:
    # Create engine
    engine = create_engine(DATABASE_URL, echo=False)
    
    # Test connection
    with engine.connect() as conn:
        print("[OK] ✅ Database connection successful!")
    
    print("")
    print("[INFO] Creating approval tables...")
    print("")
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    print("[OK] Approval tables created successfully!")
    print("")
    print("Tables created:")
    print("  - approval_links")
    print("  - consent_approvals")
    print("  - approval_history")
    print("")
    print("=" * 80)
    print("DATABASE SCHEMA INITIALIZATION COMPLETE")
    print("=" * 80)
    print("")
    
except Exception as e:
    print(f"[ERROR] Failed to create tables: {str(e)}")
    print("")
    import traceback
    traceback.print_exc()
    raise
