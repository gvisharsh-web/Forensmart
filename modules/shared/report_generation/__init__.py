"""
REPORT GENERATION MODULE - Core Package

Provides comprehensive report generation system for forensic reports with:
- Multiple report types (Executive Summary, Detailed Findings, Technical Analysis, etc.)
- Modular section generation
- Multiple export formats (Text, JSON, PDF, DOCX, HTML)
- IT Act of India compliance
- Chain of custody documentation
- Digital signatures and certification
"""

from .base_template import BaseTemplate
from .section_generator import SectionGenerator
from .formatter import ReportFormatter
from .exporter import ReportExporter
from .validator import ReportValidator

__all__ = [
    'BaseTemplate',
    'SectionGenerator',
    'ReportFormatter',
    'ReportExporter',
    'ReportValidator'
]

__version__ = '1.0.0'
__author__ = 'Forensmart Team'
__description__ = 'Advanced Report Generation System for Digital Forensics'
