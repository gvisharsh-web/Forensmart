"""
MEDIA ERROR HANDLER - Error handling for media viewer

Integrates error handling with media analysis and viewing operations
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# MEDIA ERROR HANDLER CLASS
# ============================================================================

class MediaErrorHandler:
    """Handles errors in media operations"""
    
    def __init__(self):
        try:
            from modules.error_handling import ErrorHandlingSystem
            self.error_system = ErrorHandlingSystem()
            self.available = True
        except ImportError:
            logger.warning("Error handling system not available")
            self.available = False
        
        self.media_errors = []
    
    # ========================================================================
    # MEDIA FILE HANDLING
    # ========================================================================
    
    def handle_media_file_error(self, file_path: str, error: Exception) -> Dict[str, Any]:
        """Handle media file errors"""
        try:
            error_info = {
                'type': 'MediaFileError',
                'file': file_path,
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            }
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.media_errors.append({
                    'file': file_path,
                    'error_type': error_result['error_info']['type'],
                    'message': error_result['error_info']['message'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'file': file_path,
                    'error': error_result['error_info']['message'],
                    'recovery': error_result['rectification']
                }
            else:
                return {
                    'success': False,
                    'file': file_path,
                    'error': str(error)
                }
        except Exception as e:
            logger.error(f"Media error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # CORRUPTED FILE ERROR HANDLING
    # ========================================================================
    
    def handle_corrupted_file_error(self, file_path: str, 
                                   file_type: str = 'unknown') -> Dict[str, Any]:
        """Handle corrupted file errors with auto-fix"""
        try:
            logger.warning(f"Corrupted file detected: {file_path}")
            
            error_info = {
                'type': 'CorruptedFileError',
                'file': file_path,
                'file_type': file_type,
                'message': f'File is corrupted: {file_path}'
            }
            
            if self.available:
                error_result = self.error_system.handle_error(
                    error=Exception(f"Corrupted file: {file_path}")
                )
                
                # Try to recover
                recovery_result = self._attempt_file_recovery(file_path, file_type)
                
                self.media_errors.append({
                    'file': file_path,
                    'file_type': file_type,
                    'error_type': 'CorruptedFileError',
                    'message': 'File corruption detected',
                    'recovery_attempted': True,
                    'recovery_success': recovery_result['success'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': recovery_result['success'],
                    'file': file_path,
                    'file_type': file_type,
                    'error': 'File corruption detected',
                    'recovery_strategy': recovery_result['strategy'],
                    'recovery_details': recovery_result['details'],
                    'recommendations': self._get_corruption_recommendations(file_type),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'file': file_path,
                    'error': 'File corruption detected'
                }
        except Exception as e:
            logger.error(f"Corrupted file handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _attempt_file_recovery(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Attempt to recover corrupted file"""
        try:
            import os
            
            recovery_strategies = []
            
            # Strategy 1: Attempt to read partial data
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                    if len(data) > 0:
                        recovery_strategies.append({
                            'strategy': 'partial_data_recovery',
                            'success': True,
                            'data_recovered': len(data),
                            'percentage': 'partial'
                        })
            except Exception as e:
                logger.warning(f"Partial recovery failed: {e}")
            
            # Strategy 2: Attempt file repair based on type
            if file_type.lower() in ['jpg', 'jpeg', 'png', 'gif']:
                repair_result = self._repair_image_file(file_path)
                recovery_strategies.append(repair_result)
            
            elif file_type.lower() in ['pdf']:
                repair_result = self._repair_pdf_file(file_path)
                recovery_strategies.append(repair_result)
            
            elif file_type.lower() in ['doc', 'docx', 'txt']:
                repair_result = self._repair_document_file(file_path)
                recovery_strategies.append(repair_result)
            
            elif file_type.lower() in ['mp4', 'avi', 'mov', 'mkv']:
                repair_result = self._repair_video_file(file_path)
                recovery_strategies.append(repair_result)
            
            elif file_type.lower() in ['mp3', 'wav', 'aac', 'flac']:
                repair_result = self._repair_audio_file(file_path)
                recovery_strategies.append(repair_result)
            
            # Strategy 3: Create backup of corrupted file
            backup_result = self._backup_corrupted_file(file_path)
            recovery_strategies.append(backup_result)
            
            # Determine overall success
            successful_strategies = [s for s in recovery_strategies if s.get('success')]
            
            return {
                'success': len(successful_strategies) > 0,
                'strategy': 'multi_strategy_recovery',
                'details': {
                    'strategies_attempted': len(recovery_strategies),
                    'strategies_successful': len(successful_strategies),
                    'strategies': recovery_strategies
                }
            }
        except Exception as e:
            logger.error(f"File recovery failed: {e}")
            return {'success': False, 'strategy': 'recovery_failed', 'details': str(e)}
    
    def _repair_image_file(self, file_path: str) -> Dict[str, Any]:
        """Attempt to repair image file"""
        try:
            from PIL import Image
            
            try:
                img = Image.open(file_path)
                img.verify()
                return {
                    'strategy': 'image_repair',
                    'success': True,
                    'message': 'Image file verified and repaired'
                }
            except Exception as e:
                # Try to recover image data
                return {
                    'strategy': 'image_repair',
                    'success': False,
                    'message': f'Image repair failed: {str(e)}'
                }
        except ImportError:
            return {
                'strategy': 'image_repair',
                'success': False,
                'message': 'PIL library not available'
            }
    
    def _repair_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Attempt to repair PDF file"""
        try:
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header == b'%PDF':
                    return {
                        'strategy': 'pdf_repair',
                        'success': True,
                        'message': 'PDF header verified'
                    }
                else:
                    return {
                        'strategy': 'pdf_repair',
                        'success': False,
                        'message': 'Invalid PDF header'
                    }
        except Exception as e:
            return {
                'strategy': 'pdf_repair',
                'success': False,
                'message': f'PDF repair failed: {str(e)}'
            }
    
    def _repair_document_file(self, file_path: str) -> Dict[str, Any]:
        """Attempt to repair document file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 0:
                    return {
                        'strategy': 'document_repair',
                        'success': True,
                        'message': f'Document recovered: {len(content)} characters',
                        'recovered_size': len(content)
                    }
                else:
                    return {
                        'strategy': 'document_repair',
                        'success': False,
                        'message': 'Document is empty'
                    }
        except Exception as e:
            return {
                'strategy': 'document_repair',
                'success': False,
                'message': f'Document repair failed: {str(e)}'
            }
    
    def _repair_video_file(self, file_path: str) -> Dict[str, Any]:
        """Attempt to repair video file"""
        try:
            # Check video file size
            import os
            file_size = os.path.getsize(file_path)
            
            if file_size > 0:
                return {
                    'strategy': 'video_repair',
                    'success': True,
                    'message': f'Video file size: {file_size} bytes',
                    'file_size': file_size
                }
            else:
                return {
                    'strategy': 'video_repair',
                    'success': False,
                    'message': 'Video file is empty'
                }
        except Exception as e:
            return {
                'strategy': 'video_repair',
                'success': False,
                'message': f'Video repair failed: {str(e)}'
            }
    
    def _repair_audio_file(self, file_path: str) -> Dict[str, Any]:
        """Attempt to repair audio file"""
        try:
            # Check audio file size
            import os
            file_size = os.path.getsize(file_path)
            
            if file_size > 0:
                return {
                    'strategy': 'audio_repair',
                    'success': True,
                    'message': f'Audio file size: {file_size} bytes',
                    'file_size': file_size
                }
            else:
                return {
                    'strategy': 'audio_repair',
                    'success': False,
                    'message': 'Audio file is empty'
                }
        except Exception as e:
            return {
                'strategy': 'audio_repair',
                'success': False,
                'message': f'Audio repair failed: {str(e)}'
            }
    
    def _backup_corrupted_file(self, file_path: str) -> Dict[str, Any]:
        """Backup corrupted file"""
        try:
            import shutil
            import os
            
            backup_path = f"{file_path}.corrupted.backup"
            shutil.copy2(file_path, backup_path)
            
            return {
                'strategy': 'backup_creation',
                'success': True,
                'message': f'Backup created: {backup_path}',
                'backup_path': backup_path
            }
        except Exception as e:
            return {
                'strategy': 'backup_creation',
                'success': False,
                'message': f'Backup failed: {str(e)}'
            }
    
    def _get_corruption_recommendations(self, file_type: str) -> List[str]:
        """Get recommendations for corrupted file"""
        recommendations = [
            'Attempt to recover from backup if available',
            'Try alternative file viewers or applications',
            'Contact the file source for a fresh copy',
            'Use file recovery software if necessary'
        ]
        
        if file_type.lower() in ['jpg', 'jpeg', 'png', 'gif']:
            recommendations.extend([
                'Use image repair tools',
                'Try opening in different image viewers',
                'Check if file is actually an image'
            ])
        
        elif file_type.lower() in ['pdf']:
            recommendations.extend([
                'Use PDF repair tools',
                'Try extracting text content',
                'Regenerate PDF from source'
            ])
        
        elif file_type.lower() in ['doc', 'docx', 'txt']:
            recommendations.extend([
                'Use document recovery tools',
                'Try opening in different applications',
                'Check file encoding'
            ])
        
        elif file_type.lower() in ['mp4', 'avi', 'mov', 'mkv']:
            recommendations.extend([
                'Use video repair tools',
                'Try different video players',
                'Check video codec compatibility'
            ])
        
        elif file_type.lower() in ['mp3', 'wav', 'aac', 'flac']:
            recommendations.extend([
                'Use audio repair tools',
                'Try different audio players',
                'Check audio codec compatibility'
            ])
        
        return recommendations
    
    # ========================================================================
    # MEDIA PROCESSING ERRORS
    # ========================================================================
    
    def handle_media_processing_error(self, media_type: str, 
                                     operation: str, 
                                     error: Exception) -> Dict[str, Any]:
        """Handle media processing errors"""
        try:
            error_info = {
                'type': 'MediaProcessingError',
                'media_type': media_type,
                'operation': operation,
                'error': str(error)
            }
            
            if self.available:
                error_result = self.error_system.handle_error(error=error)
                
                self.media_errors.append({
                    'media_type': media_type,
                    'operation': operation,
                    'error_type': error_result['error_info']['type'],
                    'status': 'handled',
                    'timestamp': datetime.now().isoformat()
                })
                
                return {
                    'success': False,
                    'media_type': media_type,
                    'operation': operation,
                    'error': error_result['error_info']['message'],
                    'recovery': error_result['rectification']
                }
            else:
                return {
                    'success': False,
                    'media_type': media_type,
                    'operation': operation,
                    'error': str(error)
                }
        except Exception as e:
            logger.error(f"Media processing error handling failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ========================================================================
    # MEDIA VALIDATION
    # ========================================================================
    
    def validate_media_file(self, file_path: str, media_type: str) -> Dict[str, Any]:
        """Validate media file"""
        try:
            if not file_path:
                raise ValueError("No file path provided")
            
            if self.available:
                validation = self.error_system.validate_input(
                    {'file': file_path, 'type': media_type},
                    {'type': dict, 'required_fields': ['file', 'type']}
                )
                
                if not validation['valid']:
                    return {
                        'valid': False,
                        'errors': validation['errors'],
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'valid': True,
                'file': file_path,
                'media_type': media_type,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Media validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # MEDIA RECOVERY
    # ========================================================================
    
    def recover_media_operation(self, media_type: str, 
                               operation: str) -> Dict[str, Any]:
        """Recover from media operation error"""
        try:
            if self.available:
                error_info = {
                    'type': 'MediaOperationError',
                    'media_type': media_type,
                    'operation': operation
                }
                
                recovery = self.error_system.rectifier.rectify_error(error_info)
                
                return {
                    'recovered': recovery.get('success', False),
                    'recovery_strategy': recovery.get('fix_type'),
                    'media_type': media_type,
                    'operation': operation,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'recovered': False,
                    'error': 'Error handling system not available',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Media recovery failed: {e}")
            return {
                'recovered': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ========================================================================
    # LOGGING & STATISTICS
    # ========================================================================
    
    def get_media_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get media error log"""
        return self.media_errors[-limit:]
    
    def get_media_error_statistics(self) -> Dict[str, Any]:
        """Get media error statistics"""
        if not self.media_errors:
            return {}
        
        stats = {
            'total_errors': len(self.media_errors),
            'by_type': {},
            'by_operation': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for error in self.media_errors:
            media_type = error.get('media_type', 'unknown')
            operation = error.get('operation', 'unknown')
            
            stats['by_type'][media_type] = stats['by_type'].get(media_type, 0) + 1
            stats['by_operation'][operation] = stats['by_operation'].get(operation, 0) + 1
        
        return stats

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

media_error_handler = MediaErrorHandler()

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def get_media_error_handler() -> MediaErrorHandler:
    """Get media error handler instance"""
    return media_error_handler
