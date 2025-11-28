"""
REPORT FORMATTER - Format Report Content

Provides functionality to format report content with:
- Text formatting
- Visual styling
- Layout management
- Professional appearance
"""

import logging
from typing import List, Dict, Any

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# REPORT FORMATTER CLASS
# ============================================================================

class ReportFormatter:
    """
    Format report content for professional appearance.
    
    This class provides methods to format report sections with
    proper styling, separators, and layout.
    """
    
    # Formatting constants
    HEADER_SEPARATOR = "═" * 79
    SECTION_SEPARATOR = "─" * 79
    LINE_WIDTH = 79
    
    def __init__(self):
        """Initialize report formatter"""
        logger.info("ReportFormatter initialized")
    
    # ========================================================================
    # FORMATTING METHODS
    # ========================================================================
    
    def format_header(self, title: str, level: int = 1) -> str:
        """
        Format a header with appropriate styling and visual hierarchy.
        
        Creates formatted headers with different styles based on level:
        - Level 1: Main header with top and bottom borders
        - Level 2: Section header with bottom border
        - Level 3: Subsection header without border
        
        Args:
            title (str): Header title text to format
            level (int): Header level (1, 2, or 3). Defaults to 1.
                - 1: Main header (centered with borders)
                - 2: Section header (with separator line)
                - 3: Subsection header (plain text)
            
        Returns:
            str: Formatted header string ready for inclusion in report
            
        Raises:
            Exception: If formatting fails
            
        Example:
            >>> formatter = ReportFormatter()
            >>> header1 = formatter.format_header("Main Title", level=1)
            >>> header2 = formatter.format_header("Section", level=2)
            >>> header3 = formatter.format_header("Subsection", level=3)
        """
        try:
            if level == 1:
                # Main header
                formatted = f"\n{self.HEADER_SEPARATOR}\n{title.center(self.LINE_WIDTH)}\n{self.HEADER_SEPARATOR}\n"
            elif level == 2:
                # Section header
                formatted = f"\n{title}\n{self.SECTION_SEPARATOR}\n"
            else:
                # Subsection header
                formatted = f"\n{title}\n"
            
            logger.debug(f"Header formatted: {title}")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting header: {str(e)}")
            raise
    
    def format_section(self, title: str, content: str) -> str:
        """
        Format a complete section with header and content.
        
        Combines a formatted section header with the provided content
        to create a complete, professionally formatted report section.
        
        Args:
            title (str): Section title/heading
            content (str): Section body content
            
        Returns:
            str: Complete formatted section with header and content
            
        Raises:
            Exception: If formatting fails
            
        Example:
            >>> formatter = ReportFormatter()
            >>> section = formatter.format_section(
            ...     "Executive Summary",
            ...     "This section contains the executive summary..."
            ... )
        """
        try:
            formatted = f"{self.format_header(title, level=2)}{content}\n"
            logger.debug(f"Section formatted: {title}")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting section: {str(e)}")
            raise
    
    def format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Format a table with headers and rows for professional display.
        
        Creates a formatted ASCII table with proper column alignment,
        headers, and separators. Automatically calculates column widths
        based on content.
        
        Args:
            headers (List[str]): List of column header names
            rows (List[List[str]]): List of rows, where each row is a list of cell values
            
        Returns:
            str: Formatted ASCII table ready for inclusion in report
            
        Raises:
            Exception: If formatting fails
            
        Example:
            >>> formatter = ReportFormatter()
            >>> table = formatter.format_table(
            ...     headers=["Name", "Count", "Status"],
            ...     rows=[
            ...         ["Item 1", "100", "Active"],
            ...         ["Item 2", "200", "Inactive"]
            ...     ]
            ... )
        """
        try:
            # Calculate column widths
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
            
            # Format header
            header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            separator = "-+-".join("-" * w for w in col_widths)
            
            # Format rows
            formatted_rows = []
            for row in rows:
                formatted_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
                formatted_rows.append(formatted_row)
            
            # Combine
            table = f"{header_row}\n{separator}\n" + "\n".join(formatted_rows) + "\n"
            logger.debug("Table formatted")
            return table
        
        except Exception as e:
            logger.error(f"Error formatting table: {str(e)}")
            raise
    
    def format_list(self, items: List[str], bullet_style: str = "•") -> str:
        """
        Format a bulleted list
        
        Args:
            items: List items
            bullet_style: Bullet character
            
        Returns:
            str: Formatted list
        """
        try:
            formatted = "\n".join(f"{bullet_style} {item}" for item in items)
            logger.debug("List formatted")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting list: {str(e)}")
            raise
    
    def format_key_value(self, data: Dict[str, Any], indent: int = 0) -> str:
        """
        Format key-value pairs
        
        Args:
            data: Dictionary of key-value pairs
            indent: Indentation level
            
        Returns:
            str: Formatted key-value pairs
        """
        try:
            indent_str = " " * indent
            lines = []
            
            for key, value in data.items():
                lines.append(f"{indent_str}{key}: {value}")
            
            formatted = "\n".join(lines)
            logger.debug("Key-value pairs formatted")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting key-value pairs: {str(e)}")
            raise
    
    def add_separator(self, style: str = "header") -> str:
        """
        Add a separator line
        
        Args:
            style: Separator style ('header' or 'section')
            
        Returns:
            str: Separator line
        """
        try:
            if style == "header":
                separator = self.HEADER_SEPARATOR
            else:
                separator = self.SECTION_SEPARATOR
            
            logger.debug(f"Separator added: {style}")
            return f"\n{separator}\n"
        
        except Exception as e:
            logger.error(f"Error adding separator: {str(e)}")
            raise
    
    def format_page_break(self) -> str:
        """
        Add a page break
        
        Returns:
            str: Page break marker
        """
        return "\n" + "=" * self.LINE_WIDTH + "\n[PAGE BREAK]\n" + "=" * self.LINE_WIDTH + "\n"
    
    def indent_text(self, text: str, spaces: int = 4) -> str:
        """
        Indent text
        
        Args:
            text: Text to indent
            spaces: Number of spaces to indent
            
        Returns:
            str: Indented text
        """
        try:
            indent_str = " " * spaces
            lines = text.split("\n")
            indented = "\n".join(f"{indent_str}{line}" if line else line for line in lines)
            logger.debug(f"Text indented by {spaces} spaces")
            return indented
        
        except Exception as e:
            logger.error(f"Error indenting text: {str(e)}")
            raise
    
    def center_text(self, text: str) -> str:
        """
        Center text
        
        Args:
            text: Text to center
            
        Returns:
            str: Centered text
        """
        try:
            centered = text.center(self.LINE_WIDTH)
            logger.debug("Text centered")
            return centered
        
        except Exception as e:
            logger.error(f"Error centering text: {str(e)}")
            raise
    
    def format_timestamp(self, timestamp_str: str) -> str:
        """
        Format timestamp
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            str: Formatted timestamp
        """
        try:
            # Parse and reformat timestamp
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp_str)
            formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(f"Timestamp formatted: {formatted}")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting timestamp: {str(e)}")
            return timestamp_str
    
    def format_size(self, bytes_size: int) -> str:
        """
        Format file size
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            str: Formatted size
        """
        try:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.2f} PB"
        
        except Exception as e:
            logger.error(f"Error formatting size: {str(e)}")
            return str(bytes_size)
    
    def format_percentage(self, value: float, decimals: int = 2) -> str:
        """
        Format percentage
        
        Args:
            value: Value as decimal (0-1) or percentage (0-100)
            decimals: Number of decimal places
            
        Returns:
            str: Formatted percentage
        """
        try:
            if value <= 1:
                value *= 100
            
            formatted = f"{value:.{decimals}f}%"
            logger.debug(f"Percentage formatted: {formatted}")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting percentage: {str(e)}")
            return str(value)
    
    def format_number(self, number: int, separator: str = ",") -> str:
        """
        Format number with thousands separator
        
        Args:
            number: Number to format
            separator: Thousands separator
            
        Returns:
            str: Formatted number
        """
        try:
            formatted = f"{number:,}".replace(",", separator)
            logger.debug(f"Number formatted: {formatted}")
            return formatted
        
        except Exception as e:
            logger.error(f"Error formatting number: {str(e)}")
            return str(number)
