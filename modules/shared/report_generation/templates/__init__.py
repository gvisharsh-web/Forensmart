"""
REPORT TEMPLATES PACKAGE

Provides complete report templates for different use cases:
- Executive Summary Template (1-2 pages)
- Detailed Findings Template (5-10 pages)
- Technical Analysis Template (3-5 pages)
- Risk Assessment Template (2-3 pages)
- Timeline Report Template (3-5 pages)
- IT Act India Compliant Template (15-25 pages)
- Full Comprehensive Template (20-30 pages)
"""

from .executive_summary import ExecutiveSummaryTemplate
from .detailed_findings import DetailedFindingsTemplate
from .technical_analysis import TechnicalAnalysisTemplate
from .risk_assessment import RiskAssessmentTemplate
from .timeline_report import TimelineReportTemplate
from .it_act_india import ITActIndiaTemplate
from .full_comprehensive import FullComprehensiveTemplate

__all__ = [
    'ExecutiveSummaryTemplate',
    'DetailedFindingsTemplate',
    'TechnicalAnalysisTemplate',
    'RiskAssessmentTemplate',
    'TimelineReportTemplate',
    'ITActIndiaTemplate',
    'FullComprehensiveTemplate'
]
