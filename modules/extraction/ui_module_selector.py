"""
Module Selector UI Component

Allows investigator to select which modules to extract.
User can choose from: Device Info, Communications, Location, Media, Security, Social Media
"""

import streamlit as st
from typing import Dict


def render_module_selector() -> Dict[str, bool]:
    """
    Render module selection UI.
    
    Returns:
        Dict[str, bool]: Selected modules {module_name: is_selected}
    """
    st.subheader("Step 2: Select Modules to Extract")
    
    st.write("Choose which modules to extract from the device:")
    
    # Initialize session state for modules with default values
    default_modules = {
        'device_info': True,
        'communications': True,
        'location': False,
        'media': True,
        'security': False,
        'social_media': True,
    }
    
    if 'selected_modules' not in st.session_state:
        st.session_state['selected_modules'] = default_modules.copy()
    else:
        # Ensure all keys exist (in case session state is corrupted)
        for key, value in default_modules.items():
            if key not in st.session_state['selected_modules']:
                st.session_state['selected_modules'][key] = value
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Device & System**")
        
        device_info = st.checkbox(
            "📱 Device Info",
            value=st.session_state['selected_modules'].get('device_info', True),
            help="Device model, OS version, storage, etc."
        )
        st.session_state['selected_modules']['device_info'] = device_info
        
        st.write("**Data & Content**")
        
        communications = st.checkbox(
            "💬 Communications",
            value=st.session_state['selected_modules'].get('communications', True),
            help="SMS, Messages, Emails, Call logs"
        )
        st.session_state['selected_modules']['communications'] = communications
        
        location = st.checkbox(
            "📍 Location Data",
            value=st.session_state['selected_modules'].get('location', False),
            help="GPS history, location tags, maps"
        )
        st.session_state['selected_modules']['location'] = location
    
    with col2:
        st.write("**Media & Files**")
        
        media = st.checkbox(
            "🖼️ Media",
            value=st.session_state['selected_modules'].get('media', True),
            help="Photos, Videos, Audio files"
        )
        st.session_state['selected_modules']['media'] = media
        
        st.write("**Security & Apps**")
        
        security = st.checkbox(
            "🔒 Security",
            value=st.session_state['selected_modules'].get('security', False),
            help="Installed apps, passwords, encryption"
        )
        st.session_state['selected_modules']['security'] = security
        
        social_media = st.checkbox(
            "📱 Social Media",
            value=st.session_state['selected_modules'].get('social_media', True),
            help="WhatsApp, Instagram, Telegram, Facebook, Snapchat"
        )
        st.session_state['selected_modules']['social_media'] = social_media
    
    # Show module details
    st.divider()
    
    st.write("**Module Details:**")
    
    with st.expander("📋 View Module Information"):
        st.write("""
        **Device Info**
        - Device model, manufacturer, OS version
        - Storage capacity, RAM, battery
        - IMEI, serial number
        - Network information
        
        **Communications**
        - SMS messages
        - Chat messages (WhatsApp, Telegram, etc.)
        - Email accounts and messages
        - Call logs and contacts
        
        **Location Data**
        - GPS history and coordinates
        - Location tags on photos
        - Maps and navigation history
        - Geofence data
        
        **Media**
        - Photos and images
        - Videos and recordings
        - Audio files and music
        - Screenshots and thumbnails
        
        **Security**
        - Installed applications
        - App permissions
        - Stored passwords and credentials
        - Encryption keys
        - Security settings
        
        **Social Media**
        - WhatsApp messages and media
        - Instagram direct messages and posts
        - Telegram messages and channels
        - Facebook messages and posts
        - Snapchat messages and stories
        """)
    
    # Show selected modules summary
    st.divider()
    
    selected_modules = {k: v for k, v in st.session_state['selected_modules'].items() if v}
    selected_count = len(selected_modules)
    
    if selected_count > 0:
        st.success(f"✅ {selected_count} module(s) selected")
        
        # Show selected modules
        cols = st.columns(min(3, selected_count))
        for idx, (module_name, _) in enumerate(selected_modules.items()):
            with cols[idx % 3]:
                st.write(f"✅ {format_module_name(module_name)}")
    else:
        st.warning("⚠️ Please select at least one module")
    
    # Show extraction time estimate
    show_extraction_estimate(selected_modules)
    
    return st.session_state['selected_modules']


def format_module_name(module_name: str) -> str:
    """
    Format module name for display.
    
    Args:
        module_name: Module name (snake_case)
        
    Returns:
        str: Formatted name
    """
    module_display = {
        'device_info': '📱 Device Info',
        'communications': '💬 Communications',
        'location': '📍 Location Data',
        'media': '🖼️ Media',
        'security': '🔒 Security',
        'social_media': '📱 Social Media',
    }
    
    return module_display.get(module_name, module_name)


def show_extraction_estimate(selected_modules: Dict[str, bool]) -> None:
    """
    Show estimated extraction time and data size.
    
    Args:
        selected_modules: Selected modules
    """
    if not selected_modules:
        return
    
    # Estimate based on modules
    time_estimates = {
        'device_info': 2,
        'communications': 10,
        'location': 5,
        'media': 30,
        'security': 5,
        'social_media': 15,
    }
    
    size_estimates = {
        'device_info': '50 MB',
        'communications': '200 MB',
        'location': '100 MB',
        'media': '2-5 GB',
        'security': '100 MB',
        'social_media': '500 MB - 2 GB',
    }
    
    total_time = sum(time_estimates.get(m, 0) for m in selected_modules.keys())
    
    st.divider()
    st.write("**Extraction Estimate:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Estimated Time", f"~{total_time} minutes")
    
    with col2:
        st.metric("Estimated Size", "~3-8 GB")
    
    st.info(f"⏱️ Extraction will take approximately **{total_time} minutes** depending on device speed and data volume.")


def get_module_requirements(module_name: str) -> Dict:
    """
    Get requirements for a specific module.
    
    Args:
        module_name: Module name
        
    Returns:
        Dict: Module requirements
    """
    requirements = {
        'device_info': {
            'min_consent': 'BASIC',
            'requires_root': False,
            'requires_adb': True,
            'time_estimate': 2,
        },
        'communications': {
            'min_consent': 'LEGAL',
            'requires_root': True,
            'requires_adb': True,
            'time_estimate': 10,
        },
        'location': {
            'min_consent': 'LEGAL',
            'requires_root': True,
            'requires_adb': True,
            'time_estimate': 5,
        },
        'media': {
            'min_consent': 'LEGAL',
            'requires_root': False,
            'requires_adb': True,
            'time_estimate': 30,
        },
        'security': {
            'min_consent': 'LEGAL',
            'requires_root': True,
            'requires_adb': True,
            'time_estimate': 5,
        },
        'social_media': {
            'min_consent': 'LEGAL',
            'requires_root': True,
            'requires_adb': True,
            'time_estimate': 15,
        },
    }
    
    return requirements.get(module_name, {})


def validate_module_selection(selected_modules: Dict[str, bool], consent_level: str) -> Dict:
    """
    Validate if selected modules can be extracted with given consent level.
    
    Args:
        selected_modules: Selected modules
        consent_level: Current consent level (BASIC, STANDARD, LEGAL)
        
    Returns:
        Dict: Validation result {valid: bool, issues: []}
    """
    issues = []
    
    consent_levels = {'BASIC': 1, 'STANDARD': 2, 'LEGAL': 3}
    current_level = consent_levels.get(consent_level, 0)
    
    for module_name, is_selected in selected_modules.items():
        if not is_selected:
            continue
        
        requirements = get_module_requirements(module_name)
        required_level = consent_levels.get(requirements.get('min_consent', 'BASIC'), 0)
        
        if current_level < required_level:
            issues.append(f"{format_module_name(module_name)} requires {requirements.get('min_consent')} consent")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }
