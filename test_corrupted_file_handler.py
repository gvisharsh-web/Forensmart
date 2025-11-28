#!/usr/bin/env python
"""Test corrupted file handler"""

from modules.analysis.media_error_handler import MediaErrorHandler

meh = MediaErrorHandler()

# Test 1: PDF corruption
print("TEST 1: PDF Corruption Handling")
result = meh.handle_corrupted_file_error('test.pdf', 'pdf')
print(f"  Success: {result['success']}")
print(f"  File: {result['file']}")
print(f"  File Type: {result['file_type']}")
print(f"  Recovery Strategy: {result['recovery_strategy']}")
print(f"  Strategies Attempted: {result['recovery_details']['strategies_attempted']}")
print(f"  Strategies Successful: {result['recovery_details']['strategies_successful']}")
print(f"  Recommendations: {len(result['recommendations'])} provided")
print()

# Test 2: Image corruption
print("TEST 2: Image Corruption Handling")
result = meh.handle_corrupted_file_error('test.jpg', 'jpg')
print(f"  Success: {result['success']}")
print(f"  File Type: {result['file_type']}")
print(f"  Strategies Attempted: {result['recovery_details']['strategies_attempted']}")
print(f"  Recommendations: {len(result['recommendations'])} provided")
print()

# Test 3: Document corruption
print("TEST 3: Document Corruption Handling")
result = meh.handle_corrupted_file_error('test.docx', 'docx')
print(f"  Success: {result['success']}")
print(f"  File Type: {result['file_type']}")
print(f"  Strategies Attempted: {result['recovery_details']['strategies_attempted']}")
print(f"  Recommendations: {len(result['recommendations'])} provided")
print()

# Test 4: Video corruption
print("TEST 4: Video Corruption Handling")
result = meh.handle_corrupted_file_error('test.mp4', 'mp4')
print(f"  Success: {result['success']}")
print(f"  File Type: {result['file_type']}")
print(f"  Strategies Attempted: {result['recovery_details']['strategies_attempted']}")
print(f"  Recommendations: {len(result['recommendations'])} provided")
print()

# Test 5: Audio corruption
print("TEST 5: Audio Corruption Handling")
result = meh.handle_corrupted_file_error('test.mp3', 'mp3')
print(f"  Success: {result['success']}")
print(f"  File Type: {result['file_type']}")
print(f"  Strategies Attempted: {result['recovery_details']['strategies_attempted']}")
print(f"  Recommendations: {len(result['recommendations'])} provided")
print()

# Test 6: Get error statistics
print("TEST 6: Error Statistics")
stats = meh.get_media_error_statistics()
print(f"  Total Errors: {stats.get('total_errors', 0)}")
print(f"  By Type: {stats.get('by_type', {})}")
print()

print("ALL TESTS COMPLETED SUCCESSFULLY!")
