"""
Consent Approval API Endpoints
FastAPI router for managing approval links, approvals, and history
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database operations
from ..database.consent_operations import ConsentApprovalOperations, get_db_session
from ..database.consent_approval_schema import ApprovalLink, ConsentApproval, ApprovalHistory

# Create router
router = APIRouter(prefix="/api/approvals", tags=["approvals"])

# ==================== REQUEST/RESPONSE MODELS ====================

class GenerateApprovalLinkRequest(BaseModel):
    """Request model for generating approval link"""
    case_id: str
    nominee_email: str
    consent_level: str  # STANDARD, LEGAL, FULL
    approval_method: Optional[str] = None  # PIN, PATTERN, BIOMETRIC
    expires_in_hours: Optional[int] = 24


class GenerateApprovalLinkResponse(BaseModel):
    """Response model for approval link generation"""
    approval_link: str
    token: str
    expires_at: datetime
    case_id: str
    nominee_email: str


class ApprovalLinkDetailsResponse(BaseModel):
    """Response model for approval link details"""
    case_id: str
    nominee_email: str
    consent_level: str
    approval_method: Optional[str]
    expires_at: datetime
    status: str
    created_at: datetime
    is_valid: bool


class ApproveConsentRequest(BaseModel):
    """Request model for approving consent"""
    token: str
    approval_method: str  # PIN, PATTERN, BIOMETRIC
    nominee_email: str
    pin_code: Optional[str] = None
    pattern: Optional[str] = None


class ApproveConsentResponse(BaseModel):
    """Response model for approval"""
    status: str
    approved_at: datetime
    case_id: str
    consent_level: str
    approval_method: str


class ApprovalStatusResponse(BaseModel):
    """Response model for approval status"""
    case_id: str
    status: str
    approved_at: Optional[datetime]
    consent_level: Optional[str]
    approval_method: Optional[str]
    nominee_email: Optional[str]


class ApprovalHistoryEventResponse(BaseModel):
    """Response model for history event"""
    action: str
    timestamp: datetime
    details: Optional[str]
    user_email: Optional[str]


class ApprovalHistoryResponse(BaseModel):
    """Response model for approval history"""
    case_id: str
    events: List[ApprovalHistoryEventResponse]


# ==================== DEPENDENCY ====================

def get_db() -> Session:
    """Get database session"""
    database_url = os.getenv('DATABASE_URL')
    return get_db_session(database_url)


# ==================== ENDPOINTS ====================

@router.post("/generate-link", response_model=GenerateApprovalLinkResponse)
async def generate_approval_link(
    request: GenerateApprovalLinkRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a new approval link for a nominee
    
    Args:
        request: GenerateApprovalLinkRequest
        db: Database session
    
    Returns:
        GenerateApprovalLinkResponse with approval link URL
    """
    try:
        ops = ConsentApprovalOperations(db)
        
        # Create approval link
        link = ops.create_approval_link(
            case_id=request.case_id,
            nominee_email=request.nominee_email,
            consent_level=request.consent_level,
            approval_method=request.approval_method,
            expires_in_hours=request.expires_in_hours
        )
        
        # Generate approval link URL (points to nominee approval page)
        approval_link = f"http://localhost:8501/nominee_approval?token={link.token}"
        
        return GenerateApprovalLinkResponse(
            approval_link=approval_link,
            token=link.token,
            expires_at=link.expires_at,
            case_id=link.case_id,
            nominee_email=link.nominee_email
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ops.close()


@router.get("/link/{token}", response_model=ApprovalLinkDetailsResponse)
async def get_approval_link(
    token: str,
    db: Session = Depends(get_db)
):
    """
    Get approval link details by token
    
    Args:
        token: Approval token
        db: Database session
    
    Returns:
        ApprovalLinkDetailsResponse with link details
    """
    try:
        ops = ConsentApprovalOperations(db)
        
        # Get approval link
        link = ops.get_approval_link(token)
        
        if not link:
            raise HTTPException(status_code=404, detail="Approval link not found")
        
        return ApprovalLinkDetailsResponse(
            case_id=link.case_id,
            nominee_email=link.nominee_email,
            consent_level=link.consent_level,
            approval_method=link.approval_method,
            expires_at=link.expires_at,
            status=link.status,
            created_at=link.created_at,
            is_valid=link.is_valid()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ops.close()


@router.post("/{case_id}/approve", response_model=ApproveConsentResponse)
async def approve_consent(
    case_id: str,
    request: ApproveConsentRequest,
    db: Session = Depends(get_db)
):
    """
    Approve consent using approval link
    
    Args:
        case_id: Case ID
        request: ApproveConsentRequest
        db: Database session
    
    Returns:
        ApproveConsentResponse with approval details
    """
    try:
        ops = ConsentApprovalOperations(db)
        
        # Approve consent
        approval = ops.approve_consent(
            token=request.token,
            approval_method=request.approval_method,
            nominee_email=request.nominee_email,
            pin_code=request.pin_code,
            pattern=request.pattern
        )
        
        return ApproveConsentResponse(
            status=approval.status,
            approved_at=approval.approved_at,
            case_id=approval.case_id,
            consent_level=approval.consent_level,
            approval_method=approval.approval_method
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ops.close()


@router.get("/{case_id}/status", response_model=ApprovalStatusResponse)
async def get_approval_status(
    case_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current approval status for a case
    
    Args:
        case_id: Case ID
        db: Database session
    
    Returns:
        ApprovalStatusResponse with current status
    """
    try:
        ops = ConsentApprovalOperations(db)
        
        # Get approval status
        status = ops.get_approval_status(case_id)
        
        return ApprovalStatusResponse(
            case_id=status['case_id'],
            status=status['status'],
            approved_at=status.get('approved_at'),
            consent_level=status.get('consent_level'),
            approval_method=status.get('approval_method'),
            nominee_email=status.get('nominee_email')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ops.close()


@router.get("/{case_id}/history", response_model=ApprovalHistoryResponse)
async def get_approval_history(
    case_id: str,
    db: Session = Depends(get_db)
):
    """
    Get approval history for a case
    
    Args:
        case_id: Case ID
        db: Database session
    
    Returns:
        ApprovalHistoryResponse with history timeline
    """
    try:
        ops = ConsentApprovalOperations(db)
        
        # Get approval history
        history = ops.get_approval_history(case_id)
        
        # Convert to response format
        events = [
            ApprovalHistoryEventResponse(
                action=event.action,
                timestamp=event.timestamp,
                details=event.details,
                user_email=event.user_email
            )
            for event in history
        ]
        
        return ApprovalHistoryResponse(
            case_id=case_id,
            events=events
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ops.close()


# Export router
__all__ = ['router']
