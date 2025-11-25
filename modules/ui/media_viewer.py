import streamlit as st
from PIL import Image, ImageDraw
from datetime import datetime
from modules.consent.models import ConsentLevel
from modules.shared.file_handler import file_handler
import time
from typing import List, Optional
import os
import logging

logger = logging.getLogger(__name__)


IMAGE_EXTS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico',
    '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.srw'
}
VIDEO_EXTS = {
    '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg', '.ts', '.mts', '.m2ts'
}
AUDIO_EXTS = {
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a', '.opus', '.aiff', '.au', '.ra'
}
DOCUMENT_EXTS = {
    '.pdf', '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.rtf', '.csv', '.json', '.xml'
}

SUPPORTED_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS | DOCUMENT_EXTS

try:
    from retinaface import RetinaFace  # type: ignore
    FACE_DETECTION_AVAILABLE = True
except Exception:  # retinaface may raise ValueError if tf-keras missing
    RetinaFace = None  # type: ignore
    FACE_DETECTION_AVAILABLE = False


class ForensicMediaViewer:
    def __init__(self, consent_mgr):
        self.consent = consent_mgr
        self.preview_mode = True
        self.face_detection_enabled = False

    def toggle_redaction(self, enable: bool):
        if not self.consent.verify_consent(st.session_state.case_id, ConsentLevel.BASIC):
            st.error("Consent required for redaction settings")
            return
        self.preview_mode = enable
        st.session_state.redaction_mode = "preview" if enable else "permanent"

    def toggle_face_detection(self, enable: bool):
        self.face_detection_enabled = enable
        if enable and not FACE_DETECTION_AVAILABLE:
            st.warning(
                "Face detection requires 'face_recognition' library. Install with: pip install face_recognition")
            self.face_detection_enabled = False
        elif enable and st.session_state.get("current_media"):
            self._detect_faces(st.session_state.current_media)

    def display_media(self, file_path: str):
        if not self.consent.verify_consent(st.session_state.case_id, ConsentLevel.BASIC):
            st.error("Consent required to view media")
            return

        st.session_state.current_media = file_path
        col1, col2 = st.columns([4, 1])

        image_exts = (
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif',
            '.webp', '.svg', '.ico', '.heic', '.heif', '.raw', '.cr2',
            '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.srw'
        )
        video_exts = (
            '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm',
            '.m4v', '.3gp', '.mpg', '.mpeg', '.ts', '.mts', '.m2ts'
        )
        audio_exts = (
            '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a',
            '.opus', '.aiff', '.au', '.ra'
        )
        document_exts = (
            '.pdf', '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt',
            '.pptx', '.rtf', '.csv', '.json', '.xml'
        )

        lower_path = file_path.lower()

        with col1:
            if lower_path.endswith(image_exts):
                self._show_image(file_path)
            elif lower_path.endswith(video_exts):
                self._show_video(file_path)
            elif lower_path.endswith(audio_exts):
                self._show_audio(file_path)
            elif lower_path.endswith(document_exts):
                self._show_document(file_path)
            else:
                st.warning(
                    f"Unsupported media type: {file_path.split('.')[-1].upper()}"
                )
                st.info("Supported formats: Images, Videos, Audio, Documents")

        with col2:
            preview_mode = st.toggle(
                "Preview Mode",
                value=self.preview_mode,
                key="preview_mode_toggle"
            )
            if preview_mode != self.preview_mode:
                self.toggle_redaction(preview_mode)
            
            if FACE_DETECTION_AVAILABLE:
                face_toggle = st.toggle(
                    "Auto Face Detection",
                    value=self.face_detection_enabled,
                    key="face_detection_toggle"
                )
                if face_toggle != self.face_detection_enabled:
                    self.toggle_face_detection(face_toggle)
            if st.button("Save Redacted Copy") and not self.preview_mode:
                self._save_redacted_version(file_path)

    def _get_file_icon(self, filename: str) -> str:
        """Get appropriate icon for file type"""
        filename = filename.lower()
        if any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            return '🖼️'
        elif any(filename.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv']):
            return '🎥'
        elif any(filename.endswith(ext) for ext in ['.mp3', '.wav', '.flac']):
            return '🔊'
        elif any(filename.endswith(ext) for ext in ['.pdf']):
            return '📄 PDF'
        elif any(filename.endswith(ext) for ext in ['.doc', '.docx']):
            return '📝 DOC'
        elif any(filename.endswith(ext) for ext in ['.xls', '.xlsx']):
            return '📊 XLS'
        return '📄'

    def _preview_document_content(self, file_path: str) -> str:
        """Generate a preview of document content with improved PDF handling"""
        try:
            file_path_lower = file_path.lower()
            
            # Handle PDF files with PyMuPDF for better text extraction
            if file_path_lower.endswith('.pdf'):
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    text = ""
                    # Extract text from first 2 pages
                    for page_num in range(min(2, len(doc))):
                        page_text = doc[page_num].get_text()
                        if page_text:
                            text += page_text + "\n"
                    doc.close()
                    return text[:1000] + ('...' if len(text) > 1000 else '')
                except ImportError:
                    # Fallback to PyPDF2 if PyMuPDF not available
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            text = ' '.join(page.extract_text() or '' for page in reader.pages[:2])
                            return text[:1000] + ('...' if len(text) > 1000 else '')
                    except Exception as e:
                        return f"PDF preview error (install PyMuPDF for better results): {str(e)}"
            
            # Handle other document types
            elif file_path_lower.endswith(('.doc', '.docx')):
                try:
                    import docx2txt
                    text = docx2txt.process(file_path)
                    return text[:1000] + ('...' if len(text) > 1000 else '')
                except Exception as e:
                    return f"Error reading DOCX: {str(e)}"
                
            elif file_path_lower.endswith(('.xls', '.xlsx')):
                try:
                    import pandas as pd
                    df = pd.read_excel(file_path, nrows=10)  # First 10 rows
                    return df.to_string()
                except Exception as e:
                    return f"Error reading Excel: {str(e)}"
                
            elif file_path_lower.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)
                    return content + ('...' if len(content) >= 1000 else '')
                    
        except Exception as e:
            return f"Preview error: {str(e)}"
            
        return "Preview not available for this file type"

    def _render_pdf_preview(self, file_path: str, height: int = 500) -> bool:
        """Render an embedded PDF preview using base64 encoding"""
        import base64
        
        # Check file exists and is a PDF
        if not os.path.exists(file_path) or not file_path.lower().endswith('.pdf'):
            return False
            
        try:
            # Read the PDF file as binary
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                
            # Create an iframe with the PDF viewer
            pdf_display = f"""
            <div style="width:100%; height:{height}px; overflow:hidden; border:1px solid #e0e0e0; border-radius:4px;">
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=0"
                    width="100%" 
                    height="100%" 
                    style="border:none;">
                </iframe>
            </div>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
            return True
            
        except Exception as e:
            st.error(f"Error rendering PDF: {str(e)}")
            return False

    def display_gallery(self, case_id: str, media_files: List[str]):
        """Display a gallery view of all media files for a case with pagination"""
        st.markdown(f"## 🖼️ Media Gallery - Case {case_id}")
        
        # Session state for pagination
        if 'media_page' not in st.session_state:
            st.session_state.media_page = 0
            
        ITEMS_PER_PAGE = 24  # Increased from 20 to 24 for better grid layout
        
        # Search and filter controls
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            search_term = st.text_input(
                "Search media files", 
                placeholder="Filter by name",
                key=f"media_search_{case_id}"
            )
        with col2:
            media_type = st.selectbox(
                "Media type", 
                ["All", "Images", "Videos", "Audio", "Documents"],
                key=f"media_type_{case_id}"
            )
        with col3:
            sort_order = st.selectbox(
                "Sort by",
                ["Name (A-Z)", "Name (Z-A)", "Newest first", "Oldest first"],
                key=f"sort_order_{case_id}"
            )

        # Filter and sort files
        filtered_files = []
        for file in media_files:
            file_lower = file.lower()
            filename = os.path.basename(file)
            matches_type = (
                media_type == "All"
                or (
                    media_type == "Images"
                    and any(file_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'])
                )
                or (
                    media_type == "Videos"
                    and any(file_lower.endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'])
                )
                or (
                    media_type == "Audio"
                    and any(file_lower.endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg'])
                )
                or (
                    media_type == "Documents"
                    and any(file_lower.endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.rtf'])
                )
            )
            matches_search = search_term.lower() in file_lower if search_term else True
            if matches_type and matches_search:
                # Add file with metadata for sorting
                file_info = {
                    'path': file,
                    'name': filename,
                    'ext': os.path.splitext(filename)[1].lower(),
                    'mtime': os.path.getmtime(file),
                    'size': os.path.getsize(file)
                }
                filtered_files.append(file_info)

        # Sort files
        if sort_order == "Name (A-Z)":
            filtered_files.sort(key=lambda x: x['name'].lower())
        elif sort_order == "Name (Z-A)":
            filtered_files.sort(key=lambda x: x['name'].lower(), reverse=True)
        elif sort_order == "Newest first":
            filtered_files.sort(key=lambda x: x['mtime'], reverse=True)
        elif sort_order == "Oldest first":
            filtered_files.sort(key=lambda x: x['mtime'])

        # Display gallery with pagination
        if not filtered_files:
            st.warning("No media files match the filters")
            return

        # Calculate pagination
        total_pages = (len(filtered_files) - 1) // ITEMS_PER_PAGE + 1
        current_page = st.session_state.media_page
        
        # Ensure current page is within bounds
        if current_page >= total_pages:
            current_page = total_pages - 1
            st.session_state.media_page = current_page
        
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = min((current_page + 1) * ITEMS_PER_PAGE, len(filtered_files))
        
        # Display pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if current_page > 0:
                if st.button("← Previous"):
                    st.session_state.media_page -= 1
                    st.rerun()
        with col2:
            st.caption(f"Page {current_page + 1} of {total_pages} ({len(filtered_files)} items)")
        with col3:
            if end_idx < len(filtered_files):
                if st.button("Next →"):
                    st.session_state.media_page += 1
                    st.rerun()

        # Display current page of items in a scrollable container
        with st.container():
            # Use CSS to create a scrollable container with fixed height
            st.markdown("""
            <style>
            .scrollable-container {
                max-height: 70vh;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .media-item {
                margin-bottom: 20px;
                padding: 10px;
                border-radius: 5px;
                transition: background-color 0.2s;
            }
            .media-item:hover {
                background-color: #f5f5f5;
            }
            .media-preview {
                max-height: 150px;
                max-width: 100%;
                object-fit: contain;
                margin: 0 auto;
                display: block;
            }
            .file-icon {
                font-size: 24px;
                text-align: center;
                margin: 10px 0;
            }
            .file-name {
                text-align: center;
                font-size: 0.8em;
                word-break: break-word;
                margin-top: 5px;
            }
            </style>
            <div class="scrollable-container">
                <div class="media-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px;">
            """, unsafe_allow_html=True)
            
            # Display items in a grid
            cols = st.columns(6)  # 6 columns for better space utilization
            for idx, file_info in enumerate(filtered_files[start_idx:end_idx]):
                file = file_info['path']
                filename = file_info['name']
                ext = file_info['ext']
                
                with cols[idx % 6]:
                    try:
                        # Create a button with the file icon/thumbnail
                        icon = self._get_file_icon(filename)
                        
                        # For images, show thumbnail
                        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            try:
                                img = Image.open(file)
                                img.thumbnail((150, 150))
                                st.image(img, use_container_width=True, use_column_width=True)
                            except Exception as e:
                                st.markdown(f'<div class="file-icon">{icon}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="file-icon">{icon}</div>', unsafe_allow_html=True)
                        
                        # Show filename with tooltip
                        st.markdown(f'<div class="file-name" title="{filename}">{filename[:15] + "..." if len(filename) > 15 else filename}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Preview button with enhanced functionality
                        if st.button("View", key=f"view_{idx}_{current_page}", use_container_width=True):
                            st.session_state['current_media'] = file
                            st.rerun()
                            
                        # Quick preview on hover for documents
                        if ext in ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.xls', '.xlsx']:
                            with st.expander("📄 Preview"):
                                # For PDFs, try to show embedded preview first
                                if ext == '.pdf':
                                    if not self._render_pdf_preview(file, height=400):
                                        # Fallback to text preview if PDF rendering fails
                                        preview = self._preview_document_content(file)
                                        st.text_area(
                                            "Document Preview",
                                            preview,
                                            height=150,
                                            disabled=True,
                                            key=f"doc_preview_{case_id}_{current_page}_{idx}"
                                        )
                                else:
                                    # For other document types, show text preview
                                    preview = self._preview_document_content(file)
                                    st.text_area(
                                        "Document Preview",
                                        preview,
                                        height=150,
                                        disabled=True,
                                        key=f"doc_preview_{case_id}_{current_page}_{idx}"
                                    )
                        
                    except Exception as e:
                        st.error(f"Error loading {filename}: {str(e)}")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Bottom pagination controls
        if total_pages > 1:
            st.write("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if current_page > 0:
                    if st.button("← Previous Page", key="prev_page_bottom"):
                        st.session_state.media_page -= 1
                        st.rerun()
            with col3:
                if end_idx < len(filtered_files):
                    if st.button("Next Page →", key="next_page_bottom"):
                        st.session_state.media_page += 1
                        st.rerun()
            
            # Page number selector
            with col2:
                page_options = list(range(1, total_pages + 1))
                selected_page = st.selectbox(
                    "Go to page", 
                    page_options, 
                    index=current_page,
                    key="page_selector"
                )
                if selected_page - 1 != current_page:
                    st.session_state.media_page = selected_page - 1
                    st.rerun()

    def _detect_faces(self, image_path):
        """Automatically detect faces using RetinaFace and add to redaction areas"""
        if not self.face_detection_enabled or not FACE_DETECTION_AVAILABLE:
            return

        try:
            # RetinaFace detects faces and returns bounding boxes
            faces = RetinaFace.detect_faces(image_path)

            if faces:
                st.session_state.redaction_areas = st.session_state.get(
                    "redaction_areas", []
                )
                face_count = 0

                for face_key in faces.keys():
                    face_info = faces[face_key]
                    facial_area = face_info["facial_area"]

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = facial_area
                    face_area = (x1, y1, x2, y2)

                    if face_area not in st.session_state.redaction_areas:
                        st.session_state.redaction_areas.append(face_area)
                        face_count += 1

                if face_count > 0:
                    st.success(
                        f"Auto-detected {face_count} face(s) for redaction")
                else:
                    st.info("Faces detected but already in redaction areas")
            else:
                st.info("No faces detected in image")
        except Exception as e:
            st.error(f"Face detection error: {str(e)}")

    def _show_image(self, path):
        if hasattr(self, '_show_enhanced_image'):
            return self._show_enhanced_image(path)
        try:
            img = Image.open(path)
            st.image(img, caption=os.path.basename(path))
        except Exception as exc:
            st.error(f"Image loading error: {exc}")

    def _show_video(self, path):
        if hasattr(self, '_show_enhanced_video'):
            return self._show_enhanced_video(path)
        try:
            st.video(path)
            st.caption(f"Video: {os.path.basename(path)}")
        except Exception as exc:
            st.error(f"Video playback error: {exc}")

    def _show_enhanced_image(self, path):
        """Enhanced image display with better visual presentation"""
        try:
            img = Image.open(path)

            # Apply redactions if any
            if st.session_state.get("redaction_areas"):
                img = self._apply_redactions(img)

            # Show detected faces if enabled
            if self.face_detection_enabled and st.session_state.get("redaction_areas"):
                img_with_boxes = img.copy()
                draw = ImageDraw.Draw(img_with_boxes)
                for area in st.session_state.redaction_areas:
                    draw.rectangle(area, outline="red", width=3)
                st.image(
                    img_with_boxes, caption="Faces detected (red boxes) - " + self._get_caption(path))
            else:
                st.image(img, caption=self._get_caption(path))

        except Exception as e:
            st.error(f"Image loading error: {str(e)}")
            st.info("Please ensure the image file is not corrupted")

    def _apply_redactions(self, img):
        draw = ImageDraw.Draw(img)
        for area in st.session_state.redaction_areas:
            draw.rectangle(area, fill="black")
        return img

    def _get_caption(self, path):
        mode = "PREVIEW" if self.preview_mode else "REDACTED"
        face_info = " | Faces Auto-Detected" if self.face_detection_enabled else ""
        return f"{mode}{face_info} | {path.split('/')[-1]}"

    def _save_redacted_version(self, original_path):
        img = Image.open(original_path)
        img = self._apply_redactions(img)
        redacted_path = (
            f"{original_path}_redacted_{datetime.now().timestamp()}.png"
        )
        img.save(redacted_path)
        st.success(f"Saved redacted version to: {redacted_path}")
        st.session_state.setdefault("redaction_logs", []).append({
            "original": original_path,
            "redacted": redacted_path,
            "timestamp": datetime.now(),
            "user": st.session_state.get("investigator", "Unknown"),
            "face_detection_used": self.face_detection_enabled
        })

    def _show_enhanced_video(self, path):
        """Enhanced video display with better controls"""
        st.markdown("### 🎬 Video Content")
        st.video(path)

        # Video info
        video_info = st.expander("📹 Video Information")
        with video_info:
            st.write("**Format:** " + path.split('.')[-1].upper())
            st.write("**Features:** Native playback, seeking, volume control")
            st.info("Video redaction capabilities coming in future updates")

    def _show_enhanced_audio(self, path):
        """Enhanced audio display with waveform visualization"""
        st.markdown("### 🎵 Audio Content")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.audio(path, format=f"audio/{path.split('.')[-1]}")
        with col2:
            st.metric("Format", path.split('.')[-1].upper())

        # Audio analysis
        with st.expander("🔊 Audio Analysis"):
            st.write("**File:** " + os.path.basename(path))
            st.write("**Type:** Digital audio")
            st.info("Advanced audio analysis features available in forensic modules")

    def _show_enhanced_document(self, path):
        """Enhanced document display with better formatting"""
        ext = path.lower().split('.')[-1]
        st.markdown(f"### 📄 {ext.upper()} Document")

        try:
            if ext == 'txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Enhanced text display
                st.markdown("""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;
                             font-family: 'Courier New', monospace; white-space: pre-wrap;">
                """)
                st.text_area("", content, height=400, disabled=True)
                st.markdown("</div>", unsafe_allow_html=True)

                lines = content.splitlines()
                st.metric("Lines", len(lines))
                st.metric("Characters", len(content))

            elif ext == 'json':
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                st.markdown("### 🔧 JSON Structure")
                st.json(data)

            elif ext == 'xml':
                with open(path, 'r') as f:
                    content = f.read()
                st.markdown("### 🏗️ XML Content")
                st.code(content, language='xml')

            else:
                # For complex documents, show download option
                st.info(f"📋 {ext.upper()} file detected")
                st.markdown(
                    """
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px;
                                 border-left: 4px solid #ffc107;">
                        <strong>Note:</strong> Full {ext.upper()} viewing requires additional libraries.
                        You can download the file to view it in your preferred application.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with open(path, 'rb') as f:
                    st.download_button(
                        f"⬇️ Download {ext.upper()} File",
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime=f"application/{ext}"
                    )

        except Exception as e:
            st.error(f"Document reading error: {str(e)}")
            # Fallback download
            with open(path, 'rb') as f:
                st.download_button(
                    "⬇️ Download File Anyway",
                    data=f.read(),
                    file_name=os.path.basename(path),
                    mime="application/octet-stream"
                )

    def _show_audio(self, path):
        """Display audio player for supported formats"""
        try:
            st.audio(path)
            st.caption(f"Audio: {path.split('/')[-1]}")
        except Exception as e:
            st.error(f"Audio playback error: {str(e)}")
            st.info("Some audio formats may require additional codecs")

    def _show_document(self, path):
        """Display document content for supported formats, with large PDF view"""
        ext = path.lower().split('.')[-1]

        try:
            if ext == 'pdf':
                # Large embedded PDF viewer
                if not self._render_pdf_preview(path, height=700):
                    st.info("Could not render PDF. Download or preview as text below.")
                    preview = self._preview_document_content(path)
                    st.text_area("PDF Preview (Text)", preview, height=300, disabled=True)
                st.download_button(
                    "⬇️ Download PDF",
                    data=open(path, 'rb').read(),
                    file_name=os.path.basename(path),
                    mime="application/pdf"
                )
            elif ext == 'txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                st.text_area("Document Content", content, height=400)
            elif ext == 'csv':
                import pandas as pd
                df = pd.read_csv(path)
                st.dataframe(df)
            elif ext == 'json':
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                st.json(data)
            elif ext == 'xml':
                with open(path, 'r') as f:
                    content = f.read()
                st.code(content, language='xml')
            elif ext in ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'rtf']:
                st.info(
                    f"{ext.upper()} files detected - full document viewing requires additional libraries")
                st.download_button(
                    "Download File",
                    data=open(path, 'rb').read(),
                    file_name=os.path.basename(path),
                    mime=f'application/{ext}'
                )
        except Exception as e:
            st.error(f"Document reading error: {str(e)}")
            st.download_button(
                "Download File Anyway",
                data=open(path, 'rb').read(),
                file_name=os.path.basename(path),
                mime="application/octet-stream"
            )

    def _show_image(self, file_path: str):
        """Display image with proper error handling"""
        try:
            from PIL import Image, ImageDraw
            img = Image.open(file_path)

            if st.session_state.get("redaction_areas"):
                try:
                    img = self._apply_redactions(img)
                except Exception as e:
                    logger.error(
                        f"Redaction failed: {type(e).__name__} - {str(e)}")
                    st.error(f"Redaction error: {str(e)}")

            if self.face_detection_enabled and st.session_state.get("redaction_areas"):
                try:
                    img_with_boxes = img.copy()
                    draw = ImageDraw.Draw(img_with_boxes)
                    for area in st.session_state.redaction_areas:
                        draw.rectangle(area, outline="red", width=3)
                    st.image(
                        img_with_boxes, caption=f"Detected faces - {file_path.split('/')[-1]}")
                except Exception as e:
                    logger.error(
                        f"Face detection overlay failed: {type(e).__name__} - {str(e)}")
                    st.warning("Couldn't display face detection boxes")
                    st.image(img, caption=file_path.split('/')[-1])
                else:
                    st.image(img, caption=file_path.split('/')[-1])
            else:
                st.image(img, caption=file_path.split('/')[-1])

        except Exception as e:
            logger.error(
                f"Image loading failed: {type(e).__name__} - {str(e)}")
            st.error(f"Could not load image: {str(e)}")
            st.info(f"File path: {file_path}")


def _discover_media_files(case_id: str, search_roots: Optional[List[str]] = None) -> List[str]:
    roots = search_roots or [
        os.path.join('artifacts', case_id, 'media'),
        os.path.join('artifacts', case_id),
        os.path.join('reports', case_id, 'media')
    ]

    discovered: List[str] = []
    seen = set()

    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_MEDIA_EXTS:
                    path = os.path.join(dirpath, filename)
                    if path not in seen:
                        discovered.append(path)
                        seen.add(path)
    
    st.info(f"Searching for media in: {', '.join(roots)}. Found {len(discovered)} supported media files.")

    discovered.sort()
    return discovered


def render_media_view(
    case_id: str,
    consent_manager,
    search_roots: Optional[List[str]] = None
) -> None:
    """Entry point expected by the dashboard for rendering media UI."""

    st.session_state.setdefault('case_id', case_id)
    st.session_state.setdefault('redaction_areas', [])

    media_files = _discover_media_files(case_id, search_roots=search_roots)
    if not media_files:
        st.info('No supported media artifacts found for this case.')
        return

    viewer = ForensicMediaViewer(consent_manager)
    viewer.display_gallery(case_id, media_files)

    current_media = st.session_state.get('current_media')
    if not current_media or current_media not in media_files:
        st.session_state['current_media'] = media_files[0]

    viewer.display_media(st.session_state['current_media'])
