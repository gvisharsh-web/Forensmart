# Test D: End-to-End Flow - Checklist

## Status: READY TO TEST ✅

### Dashboard Running:
- ✅ Local URL: http://localhost:8501
- ✅ Network URL: http://10.14.0.112:8501
- ✅ ADB Device Connected: SCYLX46LKRS8WCIF

---

## Test Steps (Follow in Order)

### STEP 1: Create a Test Case
- [ ] Open: http://10.14.0.112:8501
- [ ] Go to: **Case Management** tab
- [ ] Click: **Create New Case**
- [ ] Fill in:
  - Subject Name: `Test Subject`
  - Device ID: `TEST_DEVICE_001`
  - Consent Level: `STANDARD`
- [ ] Click: **Create Case**
- [ ] ✅ Verify: Case created (e.g., `CASE-Test_Subject-20251122191000`)

**Expected Result:**
```
✅ Case appears in sidebar
✅ Case ID shown in dropdown
```

---

### STEP 2: Navigate to Consent Hub
- [ ] Go to: **Consent Hub** tab
- [ ] Verify: Case is selected in the tab

**Expected Result:**
```
✅ "Managing Consent for: CASE-Test_Subject-..."
```

---

### STEP 3: Generate Approval Link & Hash
- [ ] Fill in form:
  - Nominee Name: `Test Nominee`
  - Nominee Phone: `+919876543210` (or your test phone)
  - Nominee Email: `test@example.com`
  - Purpose: `Test Extraction`
- [ ] Click: **Generate Approval Link & Fallback Hash**
- [ ] ✅ Verify: Link and hash generated

**Expected Result:**
```
✅ Approval link displayed (starts with http://10.14.0.112:8501?data=...)
✅ Fallback hash displayed (8 uppercase letters, e.g., A7B9C1D2)
✅ Sharing options shown (QR, WhatsApp, SMS, Email)
```

---

### STEP 4A: Test Approval via Link (PRIMARY METHOD)

#### On Another Device/Browser (Simulate Nominee):
- [ ] Copy the approval link
- [ ] Open link in new browser/device
- [ ] ✅ Verify: See ONLY approval form (no dashboard tabs)
- [ ] ✅ Verify: Case information displayed
- [ ] Click: **✅ Yes, Approve**
- [ ] ✅ Verify: Success message "Approval Granted!"
- [ ] ✅ Verify: Balloons animation
- [ ] ✅ Verify: "This page will close automatically in 3 seconds..."
- [ ] Wait: Page auto-closes

**Expected Result:**
```
✅ Nominee sees ONLY consent form
✅ No dashboard tabs visible
✅ Page auto-closes after 3 seconds
```

#### On Investigator's Dashboard:
- [ ] Stay on **Consent Hub** tab
- [ ] Watch: **Live Approval Status** section
- [ ] ✅ Verify: Status updates to "✅ Approved by Test Nominee at [time]"
- [ ] ✅ Verify: Balloons animation
- [ ] ✅ Verify: Message about extraction starting

**Expected Result:**
```
✅ Approval detected in real-time
✅ Status shows nominee name and timestamp
✅ Dashboard ready for extraction
```

---

### STEP 4B: Test SMS Hash Fallback (FALLBACK METHOD)

#### Option 1: Manual Entry (No ADB needed)
- [ ] In **Consent Hub**, find **SMS Hash Fallback** section
- [ ] Left column: **Manual Entry**
- [ ] Copy the fallback hash (e.g., `A7B9C1D2`)
- [ ] Paste in text field: "Enter Fallback Hash from SMS"
- [ ] Click: **Verify Hash and Start Extraction**
- [ ] ✅ Verify: Success message "SMS Fallback Approval successful!"

**Expected Result:**
```
✅ Hash verified
✅ Extraction triggered
✅ Success message shown
```

#### Option 2: Auto-Read from ADB (If phone connected)
- [ ] In **Consent Hub**, find **SMS Hash Fallback** section
- [ ] Right column: **Auto-Read from ADB**
- [ ] ✅ Verify: "✅ ADB Device Connected"
- [ ] On connected Android phone, send SMS:
  ```
  APPROVE A7B9C1D2
  ```
- [ ] Click: **🔍 Read SMS from Nominee Phone**
- [ ] ✅ Verify: System reads SMS
- [ ] ✅ Verify: Hash extracted
- [ ] ✅ Verify: Success message "SMS Auto-Read Successful!"

**Expected Result:**
```
✅ ADB device detected
✅ SMS read from phone
✅ Hash extracted automatically
✅ Extraction triggered
```

---

### STEP 5: Verify Extraction Starts
- [ ] After approval (link or SMS hash):
- [ ] ✅ Verify: Dashboard auto-navigates to **Extraction** tab
- [ ] ✅ Verify: Case ID is loaded
- [ ] ✅ Verify: Device status shown (Connected/Offline)
- [ ] ✅ Verify: Battery and Storage metrics visible
- [ ] ✅ Verify: Extraction type shown (Android/iOS/HDD)

**Expected Result:**
```
✅ Extraction tab selected automatically
✅ Case information loaded
✅ Device status visible
✅ Ready to start extraction
```

---

### STEP 6: Verify Audit Trail
- [ ] Go to: **Diagnostics** tab
- [ ] Find: **Approval Status** section
- [ ] ✅ Verify: "Approved: ✅ Yes"
- [ ] ✅ Verify: "Expired: ✅ No"
- [ ] ✅ Verify: Timestamp visible

**Expected Result:**
```
✅ Approval status recorded
✅ Timestamp correct
✅ Not expired
```

---

## Test Results Summary

### Scenario A: Approval via Link ✅
- [ ] Link generated with Network IP
- [ ] Nominee sees ONLY approval form
- [ ] Page auto-closes
- [ ] Dashboard detects approval
- [ ] Extraction tab auto-selected
- [ ] Audit trail recorded

### Scenario B: SMS Hash Manual ✅
- [ ] Hash generated
- [ ] Investigator enters hash
- [ ] Hash verified
- [ ] Extraction triggered
- [ ] Audit trail recorded

### Scenario C: SMS Hash Auto-Read ✅
- [ ] Nominee sends SMS
- [ ] ADB reads SMS
- [ ] Hash extracted
- [ ] Hash verified
- [ ] Extraction triggered
- [ ] Audit trail recorded

---

## Issues Found

| Issue | Status | Fix |
|-------|--------|-----|
| | | |
| | | |
| | | |

---

## Overall Result

- [ ] All tests passed ✅
- [ ] Some tests failed ⚠️
- [ ] Critical issues found ❌

**Notes:**
```
[Write any notes here]
```

---

## Next Steps

If all tests pass:
1. ✅ Commit to Git
2. ✅ Create release notes
3. ✅ Deploy to production

If issues found:
1. ⚠️ Document issues
2. ⚠️ Fix issues
3. ⚠️ Re-test
4. ⚠️ Then commit to Git
