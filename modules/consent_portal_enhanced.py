"""Enhanced consent portal with QR codes and link delivery."""
from __future__ import annotations

import logging
import json
import base64
from typing import Dict, Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


class ConsentPortalEnhancer:
    """Enhance consent portal with QR codes and delivery options."""

    @staticmethod
    def generate_qr_code_url(approval_link: str) -> str:
        """Generate QR code URL for approval link."""
        try:
            # Use QR code API (free service)
            qr_api = "https://api.qrserver.com/v1/create-qr-code/"
            params = f"?size=300x300&data={quote(approval_link)}"
            qr_url = qr_api + params
            logger.info(f"Generated QR code URL for approval link")
            return qr_url
        except Exception as e:
            logger.error(f"Failed to generate QR code: {e}")
            return ""

    @staticmethod
    def create_whatsapp_link(phone: str, approval_link: str, nominee_name: str = "") -> str:
        """Create WhatsApp share link."""
        try:
            message = (
                f"Hi {nominee_name or 'there'},\n\n"
                f"Please review and approve this ForenSmart extraction request:\n\n"
                f"{approval_link}\n\n"
                f"Thank you!"
            )
            encoded_message = quote(message)
            whatsapp_link = f"https://wa.me/{phone}?text={encoded_message}"
            logger.info(f"Created WhatsApp link for {phone}")
            return whatsapp_link
        except Exception as e:
            logger.error(f"Failed to create WhatsApp link: {e}")
            return ""

    @staticmethod
    def create_sms_link(phone: str, approval_link: str) -> str:
        """Create SMS share link."""
        try:
            message = f"ForenSmart approval link: {approval_link}"
            encoded_message = quote(message)
            sms_link = f"sms:{phone}?body={encoded_message}"
            logger.info(f"Created SMS link for {phone}")
            return sms_link
        except Exception as e:
            logger.error(f"Failed to create SMS link: {e}")
            return ""

    @staticmethod
    def create_email_link(email: str, approval_link: str, case_id: str = "") -> str:
        """Create email share link."""
        try:
            subject = f"ForenSmart Extraction Approval Request - {case_id}"
            body = (
                f"Please review and approve this ForenSmart extraction request:\n\n"
                f"{approval_link}\n\n"
                f"Thank you!"
            )
            encoded_subject = quote(subject)
            encoded_body = quote(body)
            email_link = f"mailto:{email}?subject={encoded_subject}&body={encoded_body}"
            logger.info(f"Created email link for {email}")
            return email_link
        except Exception as e:
            logger.error(f"Failed to create email link: {e}")
            return ""

    @staticmethod
    def add_link_expiration(approval_link: str, hours: int = 24) -> str:
        """Add expiration info to approval link."""
        try:
            from datetime import datetime, timedelta
            
            expiry_time = datetime.now() + timedelta(hours=hours)
            expiry_str = expiry_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Append expiration as fragment (not sent to server)
            link_with_expiry = f"{approval_link}#expires={quote(expiry_str)}"
            logger.info(f"Added {hours}h expiration to approval link")
            return link_with_expiry
        except Exception as e:
            logger.error(f"Failed to add expiration: {e}")
            return approval_link

    @staticmethod
    def create_approval_details_json(
        case_id: str,
        device_id: str,
        purpose: str,
        requested_level: str,
        nominee_name: str = ""
    ) -> str:
        """Create JSON with approval details for display."""
        try:
            details = {
                "case_id": case_id,
                "device_id": device_id,
                "purpose": purpose,
                "requested_level": requested_level,
                "nominee_name": nominee_name,
                "created_at": datetime.now().isoformat(),
            }
            
            json_str = json.dumps(details)
            encoded = base64.b64encode(json_str.encode()).decode()
            logger.info("Created approval details JSON")
            return encoded
        except Exception as e:
            logger.error(f"Failed to create approval details: {e}")
            return ""

    @staticmethod
    def get_delivery_options(
        approval_link: str,
        nominee_phone: str = "",
        nominee_email: str = "",
        nominee_name: str = "",
        case_id: str = ""
    ) -> Dict[str, Dict[str, str]]:
        """Get all available delivery options."""
        options = {
            "direct_link": {
                "label": "Direct Link",
                "url": approval_link,
                "description": "Copy and share directly"
            },
            "qr_code": {
                "label": "QR Code",
                "url": ConsentPortalEnhancer.generate_qr_code_url(approval_link),
                "description": "Scan with phone camera"
            }
        }

        if nominee_phone:
            options["whatsapp"] = {
                "label": "WhatsApp",
                "url": ConsentPortalEnhancer.create_whatsapp_link(
                    nominee_phone, approval_link, nominee_name
                ),
                "description": "Send via WhatsApp"
            }
            options["sms"] = {
                "label": "SMS",
                "url": ConsentPortalEnhancer.create_sms_link(nominee_phone, approval_link),
                "description": "Send via SMS"
            }

        if nominee_email:
            options["email"] = {
                "label": "Email",
                "url": ConsentPortalEnhancer.create_email_link(
                    nominee_email, approval_link, case_id
                ),
                "description": "Send via Email"
            }

        return options

    @staticmethod
    def render_delivery_ui(
        approval_link: str,
        nominee_phone: str = "",
        nominee_email: str = "",
        nominee_name: str = "",
        case_id: str = ""
    ) -> None:
        """Render delivery options UI in Streamlit."""
        try:
            import streamlit as st
        except ImportError:
            logger.warning("Streamlit not available for UI rendering")
            return

        st.markdown("### 📤 Share Approval Link")
        
        options = ConsentPortalEnhancer.get_delivery_options(
            approval_link, nominee_phone, nominee_email, nominee_name, case_id
        )

        cols = st.columns(len(options))
        
        for col, (key, option) in zip(cols, options.items()):
            with col:
                if st.button(f"📱 {option['label']}", key=f"delivery_{key}"):
                    if key == "direct_link":
                        st.code(option['url'])
                        st.info("Copy the link above and share it")
                    elif key == "qr_code":
                        st.image(option['url'], caption="Scan with phone camera")
                    else:
                        st.markdown(f"[{option['label']}]({option['url']})")
                        st.caption(option['description'])


from datetime import datetime

__all__ = ["ConsentPortalEnhancer"]
