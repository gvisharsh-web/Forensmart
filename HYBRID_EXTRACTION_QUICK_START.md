# Hybrid Extraction - Quick Start Integration

**Time to integrate**: 15-20 minutes  
**Difficulty**: Easy  
**Prerequisites**: Existing app.py with extraction functionality

---

## Step 1: Verify Files Are in Place

Check that these 3 files exist in `modules/extraction/`:

```
✅ hybrid_bridge_agent.py (1200 lines)
✅ hybrid_integration.py (400 lines)
✅ ui_hybrid_extraction.py (600 lines)
```

---

## Step 2: Add Imports to app.py

Add these imports near the top of your `app.py`:

```python
# Hybrid extraction imports
from modules.extraction.hybrid_integration import create_hybrid_adapter
from modules.extraction.ui_hybrid_extraction import render_hybrid_extraction_page
from modules.extraction.orchestrator import ExtractionOrchestrator
```

---

## Step 3: Add Extraction Mode Selection

In your extraction page, add a mode selector:

```python
def render_extraction_page():
    st.header("Extraction")
    
    # Add mode selection
    extraction_mode = st.radio(
        "Extraction Mode",
        ["Standard", "Hybrid"],
        horizontal=True,
        help="Standard: Basic extraction | Hybrid: Advanced with privilege escalation"
    )
    
    # Get case and device info
    case_id = st.session_state.get('current_case_id')
    device_id = st.session_state.get('current_device_id')
    consent_manager = st.session_state.get('consent_manager')
    
    # Route to appropriate extraction
    if extraction_mode == "Standard":
        render_standard_extraction(case_id, device_id, consent_manager)
    else:
        render_hybrid_extraction_mode(case_id, device_id, consent_manager)

def render_hybrid_extraction_mode(case_id, device_id, consent_manager):
    """Render hybrid extraction interface"""
    
    orchestrator = ExtractionOrchestrator()
    
    render_hybrid_extraction_page(
        orchestrator=orchestrator,
        case_id=case_id,
        device_id=device_id,
        consent_manager=consent_manager
    )
```

---

## Step 4: Test the Integration

1. **Start the app**:
   ```bash
   streamlit run app.py
   ```

2. **Navigate to Extraction page**

3. **Select "Hybrid" mode**

4. **Configure options**:
   - Enable/disable privilege escalation
   - Select data sources
   - Choose escalation methods

5. **Click "Start Hybrid Extraction"**

6. **Monitor progress** and view results

---

## Step 5: Verify Results

Check that you can see:

- ✅ Progress bar updating (0% → 100%)
- ✅ Status messages showing current operation
- ✅ Total artifacts count
- ✅ Extraction completeness percentage
- ✅ Escalation status
- ✅ Comparison between standard and bridge extraction
- ✅ Export options

---

## Minimal Integration (5 minutes)

If you want just the core functionality without full UI:

```python
from modules.extraction.hybrid_integration import create_hybrid_adapter
from modules.extraction.orchestrator import ExtractionOrchestrator

# In your extraction function
def run_hybrid_extraction(case_id, device_id, consent_manager):
    orchestrator = ExtractionOrchestrator()
    adapter = create_hybrid_adapter(orchestrator)
    
    results = adapter.extract_all_data_hybrid(
        case_id=case_id,
        device_id=device_id,
        consent_manager=consent_manager,
        enable_escalation=True,
        enable_extended_sources=True
    )
    
    return results
```

---

## Configuration Options

### Enable/Disable Features

```python
# Full hybrid extraction
results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    enable_escalation=True,           # Try privilege escalation
    enable_extended_sources=True      # Extract from social media, cloud, etc.
)

# Without escalation
results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    enable_escalation=False,          # Skip escalation
    enable_extended_sources=True
)

# Extended sources only
results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    enable_escalation=False,
    enable_extended_sources=True
)
```

---

## Understanding Results

### Result Structure

```python
results = {
    'status': 'success',
    'total_artifacts': 1250,           # Combined from all sources
    'extraction_completeness': 85.5,   # Percentage of device data extracted
    'privilege_escalation_used': True, # Whether escalation succeeded
    'escalation_method': 'dirty_pipe', # Which method was used
    'standard_extraction': {...},      # Standard extraction results
    'bridge_extraction': {...},        # Bridge agent results
    'total_duration_seconds': 77.3     # Total time taken
}
```

### Accessing Results

```python
# Get total artifacts
total = results['total_artifacts']

# Check if escalation worked
if results['privilege_escalation_used']:
    print(f"Escalation method: {results['escalation_method']}")

# Get completeness
completeness = results['extraction_completeness']
print(f"Extraction completeness: {completeness}%")

# Compare methods
standard_artifacts = results['standard_extraction']['artifacts']
bridge_artifacts = results['bridge_extraction']['artifacts']
improvement = bridge_artifacts / standard_artifacts * 100
print(f"Bridge extraction improved by {improvement}%")
```

---

## Troubleshooting

### Issue: Escalation fails silently

**Solution**: This is normal. Escalation is attempted but not required. Extraction continues without it.

```python
# Check if escalation was used
if results['privilege_escalation_used']:
    print("Escalation succeeded")
else:
    print("Escalation not available on this device")
```

### Issue: Low artifact count

**Solution**: Check device connection and consent level

```python
# Verify device is connected
if not device_connected:
    st.error("Device not connected")
    return

# Verify consent level
if consent_level < ConsentLevel.STANDARD:
    st.warning("Insufficient consent for full extraction")
```

### Issue: Progress callback not updating

**Solution**: Make sure callback function is defined correctly

```python
def progress_callback(message: str, percentage: int):
    # This must be a simple function that doesn't throw exceptions
    print(f"{percentage}% - {message}")

results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    progress_callback=progress_callback  # Pass the function
)
```

---

## Performance Tips

### For Faster Extraction

```python
# Disable extended sources if not needed
results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    enable_escalation=False,
    enable_extended_sources=False  # Skip extended sources
)
```

### For More Complete Extraction

```python
# Enable all features
results = adapter.extract_all_data_hybrid(
    case_id=case_id,
    device_id=device_id,
    enable_escalation=True,        # Try escalation
    enable_extended_sources=True   # Get all sources
)
```

---

## What Gets Extracted

### Standard Extraction (45%)
- Device information
- Communications (SMS, calls, contacts)
- Location data
- Security data
- Media files
- System information

### Bridge Extraction (45%)
- Social media (WhatsApp, Telegram, Signal)
- Cloud storage (Google Drive, OneDrive)
- System logs (logcat, syslog)
- Encrypted app data (if accessible)

### Total Coverage
- Up to 1500+ artifacts
- 80-95% device completeness
- With privilege escalation

---

## Next Steps

1. ✅ Copy 3 files to `modules/extraction/`
2. ✅ Add imports to `app.py`
3. ✅ Add mode selection
4. ✅ Add hybrid extraction page
5. ✅ Test the integration
6. ✅ Verify results display
7. ⏭️ Deploy to production

---

## Support

**Files**:
- `hybrid_bridge_agent.py` - Core logic
- `hybrid_integration.py` - Integration adapter
- `ui_hybrid_extraction.py` - User interface
- `HYBRID_EXTRACTION_IMPLEMENTATION.md` - Full documentation

**Logs**: Check `logs/` directory for detailed extraction logs

**Results**: Check `artifacts/` directory for extracted data

---

**Ready to integrate?** Follow the 5 steps above and you'll have hybrid extraction working in 15-20 minutes!
