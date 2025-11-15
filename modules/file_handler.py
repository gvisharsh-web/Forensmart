import streamlit as st
import subprocess
import sys
import time
import os
from typing import Dict, Optional
from datetime import datetime


class IntelligentFileHandler:
    """Shared file handling utility for all ForenSmart modules"""

    def __init__(self):
        self.extension_database = self._build_extension_database()

    def _build_extension_database(self) -> Dict[str, Dict]:
        """Comprehensive mapping of file extensions to required packages"""
        return {
            # Media formats
            **dict.fromkeys(['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                          {'category': 'image', 'package': 'Pillow', 'safe': True, 'builtin': True}),
            **dict.fromkeys(['mp4', 'mov', 'avi', 'mkv'],
                          {'category': 'video', 'package': 'opencv-python', 'safe': True}),
            **dict.fromkeys(['mp3', 'wav', 'flac'],
                          {'category': 'audio', 'package': 'pydub', 'safe': True}),

            # Data formats
            **dict.fromkeys(['csv', 'xlsx', 'xls'],
                          {'category': 'data', 'package': 'pandas', 'safe': True}),
            **dict.fromkeys(['json', 'xml'],
                          {'category': 'data', 'package': 'builtins', 'safe': True, 'builtin': True}),

            # Forensic formats
            **dict.fromkeys(['zip', 'rar', '7z'],
                          {'category': 'archive', 'package': 'zipfile', 'safe': True}),
            **dict.fromkeys(['dd', 'e01'],
                          {'category': 'forensic', 'package': 'construct', 'safe': True}),

            # Specialized
            **dict.fromkeys(['gpx', 'kml'],
                          {'category': 'gis', 'package': 'geopandas', 'safe': True}),
            **dict.fromkeys(['db', 'sqlite'],
                          {'category': 'database', 'package': 'sqlite3', 'safe': True, 'builtin': True}),
        }

    def handle_unsupported_file(self, file_path: str, module_name: str = "") -> None:
        """Handle unsupported file types with intelligent suggestions"""
        file_info = self.analyze_file(file_path)
        ext = file_info['extension']

        st.warning(f"🔍 {module_name} detected: {ext.upper()} file")

        if file_info['supported']:
            support_info = file_info

            with st.expander(f"💡 Enable {ext.upper()} Support"):
                st.write(f"**Package**: `{support_info['package']}`")

                if support_info.get('builtin'):
                    st.success("✅ Built-in support - no installation needed!")
                    return

                col1, col2 = st.columns(2)

                with col1:
                    if support_info.get('safe', False):
                        if st.button(f"🚀 Install {support_info['package']}"):
                            self._install_package(
                                support_info['package'], file_path)

                with col2:
                    if st.button("📋 Show Command"):
                        st.code(f"pip install {support_info['package']}")

        # Download fallback
        try:
            with open(file_path, 'rb') as f:
                st.download_button(
                    "⬇️ Download File",
                    data=f.read(),
                    file_name=file_info['filename'],
                    mime="application/octet-stream"
                )
        except Exception:
            pass

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze a file and return handling information"""
        ext = file_path.lower().split('.')[-1]

        return {
            'extension': ext,
            'supported': ext in self.extension_database,
            'path': file_path,
            'filename': file_path.split('/')[-1],
            **self.extension_database.get(ext, {})
        }

    def _install_package(self, package_name: str, file_path: str) -> None:
        """Attempt safe package installation"""
        with st.spinner(f"Installing {package_name}..."):
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package_name
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    st.success(f"✅ Installed {package_name}!")
                    st.info("🔄 Refreshing...")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Installation failed")
                    st.code(f"pip install {package_name}")

            except Exception as e:
                st.error(f"💥 Error: {str(e)}")
                st.code(f"pip install {package_name}")

    def register_custom_format(self, extension: str, package: str,
                              category: str, safe: bool = True) -> None:
        """Allow modules to register custom formats"""
        self.extension_database[extension.lower()] = {
            'category': category,
            'package': package,
            'safe': safe,
            'builtin': False
        }


# Global instance
file_handler = IntelligentFileHandler()
