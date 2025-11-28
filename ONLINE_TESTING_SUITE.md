# 🌐 ONLINE TESTING SUITE

**Version**: 1.0  
**Date**: November 28, 2025  
**Status**: 📋 Testing Plan for Streamlit Cloud

---

## 📊 OVERVIEW

Complete testing suite for ForenSmart when deployed online to Streamlit Cloud:
- ✅ Web app functionality testing
- ✅ Device detection testing
- ✅ Case management testing
- ✅ Extraction workflow testing
- ✅ Consent approval testing
- ✅ Token generation testing
- ✅ Desktop tool integration testing
- ✅ End-to-end workflow testing

---

## 🎯 TESTING STRATEGY

### **What We Test Online**
1. ✅ Web app loads correctly
2. ✅ Case creation works
3. ✅ Device detection works
4. ✅ Token generation works
5. ✅ Desktop tool integration works
6. ✅ Extraction workflow works
7. ✅ Results display works
8. ✅ Error handling works

### **What We Don't Test Online**
- ❌ Real device extraction (use desktop tool)
- ❌ USB device access (browser limitation)
- ❌ ADB commands directly (use desktop tool)

---

## 🧪 TEST SUITE 1: WEB APP FUNCTIONALITY

### **Test 1.1: App Loads**
```
Steps:
1. Go to: https://forensmart.streamlit.app
2. Wait for app to load
3. Check: Page loads without errors

Expected:
✅ App loads
✅ Dashboard visible
✅ All tabs accessible
```

### **Test 1.2: Navigation Works**
```
Steps:
1. Click each tab in sidebar
2. Check each page loads

Expected:
✅ Cases tab loads
✅ Extraction tab loads
✅ Reports tab loads
✅ Analysis tab loads
✅ Automation tab loads
```

### **Test 1.3: Session State Persists**
```
Steps:
1. Create a case
2. Refresh page
3. Check case still exists

Expected:
✅ Case persists after refresh
✅ Session state maintained
```

---

## 🧪 TEST SUITE 2: CASE MANAGEMENT

### **Test 2.1: Create Case**
```
Steps:
1. Go to Cases tab
2. Click "Create New"
3. Fill in:
   - Case Name: "Test Case 001"
   - Investigator: "Test Investigator"
   - Device ID: "AUTO-GENERATE"
   - Description: "Test case for online testing"
4. Click "Create Case"

Expected:
✅ Case created successfully
✅ Case ID generated
✅ Case appears in list
✅ Success message shown
```

### **Test 2.2: View Cases**
```
Steps:
1. Go to Cases tab
2. Click "All Cases"
3. Check case list

Expected:
✅ Created case visible
✅ Case details shown
✅ Case ID correct
✅ Case status correct
```

### **Test 2.3: Case Details**
```
Steps:
1. Click on case in list
2. View case details

Expected:
✅ Case name displayed
✅ Investigator name displayed
✅ Device ID displayed
✅ Creation date shown
✅ Status shown
```

---

## 🧪 TEST SUITE 3: DEVICE DETECTION

### **Test 3.1: Device Selection Options**
```
Steps:
1. Go to Extraction tab
2. Step 1: Device Selection
3. Check radio buttons

Expected:
✅ "From Connected Devices" option visible
✅ "From Your Cases" option visible
✅ "Manual Entry" option visible
```

### **Test 3.2: From Your Cases**
```
Steps:
1. Go to Extraction tab
2. Select "From Your Cases"
3. Select a case

Expected:
✅ Case list shows
✅ Can select case
✅ Case ID selected
✅ Device selected
```

### **Test 3.3: Manual Entry**
```
Steps:
1. Go to Extraction tab
2. Select "Manual Entry"
3. Enter device ID: "TEST-DEVICE-001"

Expected:
✅ Text input visible
✅ Can enter device ID
✅ Device selected
```

### **Test 3.4: From Connected Devices**
```
Steps:
1. Go to Extraction tab
2. Select "From Connected Devices"
3. Check for devices

Expected:
✅ Detection message shown
✅ Either devices found or message shown
✅ No errors
```

---

## 🧪 TEST SUITE 4: TOKEN GENERATION

### **Test 4.1: Generate Token**
```
Steps:
1. Go to Extraction tab
2. Step 3: Consent Check
3. Look for token generation option

Expected:
✅ Token generation available
✅ Can generate token
✅ Token displayed
```

### **Test 4.2: Copy Token**
```
Steps:
1. Generate token
2. Click "Copy Token"
3. Check clipboard

Expected:
✅ Token copied
✅ Token format correct
✅ Token starts with "FORENSMART_CONSENT_TOKEN_v1.0"
```

### **Test 4.3: Download Token**
```
Steps:
1. Generate token
2. Click "Download JSON"
3. Check file downloaded

Expected:
✅ JSON file downloaded
✅ File contains token data
✅ File has correct format
```

---

## 🧪 TEST SUITE 5: EXTRACTION WORKFLOW

### **Test 5.1: Module Selection**
```
Steps:
1. Go to Extraction tab
2. Step 1: Select device
3. Step 2: Module Selection
4. Check modules

Expected:
✅ Module list visible
✅ Can select modules
✅ Checkboxes work
```

### **Test 5.2: Consent Verification**
```
Steps:
1. Go to Extraction tab
2. Step 3: Consent Check
3. Check consent options

Expected:
✅ Consent level shown
✅ Allowed modules shown
✅ Blocked modules shown
```

### **Test 5.3: Extraction Progress**
```
Steps:
1. Go to Extraction tab
2. Step 4: Extraction Progress
3. Click "Start Extraction"

Expected:
✅ Progress bar appears
✅ Status updates
✅ Extraction completes
```

### **Test 5.4: Results Display**
```
Steps:
1. Go to Extraction tab
2. Step 5: Results
3. Check results

Expected:
✅ Results displayed
✅ File count shown
✅ Size shown
✅ Module status shown
```

---

## 🧪 TEST SUITE 6: DESKTOP TOOL INTEGRATION

### **Test 6.1: Token from Web App**
```
Steps:
1. Generate token in web app
2. Copy token
3. Run desktop tool: python desktop_extraction_tool.py
4. Paste token

Expected:
✅ Token accepted
✅ Token verified
✅ Consent data extracted
```

### **Test 6.2: Device Detection in Desktop Tool**
```
Steps:
1. Run desktop tool
2. Paste token
3. Check device detection

Expected:
✅ Devices detected
✅ Can select device
✅ Device confirmed
```

### **Test 6.3: Extraction in Desktop Tool**
```
Steps:
1. Run desktop tool
2. Paste token
3. Select device
4. Start extraction

Expected:
✅ Extraction starts
✅ Progress shown
✅ Extraction completes
```

### **Test 6.4: Results Upload**
```
Steps:
1. Desktop tool completes extraction
2. Results upload to web app
3. Check web app for results

Expected:
✅ Results uploaded
✅ Results visible in web app
✅ Case status updated
```

---

## 🧪 TEST SUITE 7: ERROR HANDLING

### **Test 7.1: Invalid Token**
```
Steps:
1. Desktop tool
2. Paste invalid token
3. Check error handling

Expected:
✅ Error message shown
✅ Clear error description
✅ No crash
```

### **Test 7.2: Expired Token**
```
Steps:
1. Create old token (manually set expiry to past)
2. Try to use in desktop tool
3. Check error handling

Expected:
✅ Expiry error shown
✅ Clear message
✅ No crash
```

### **Test 7.3: Missing Device**
```
Steps:
1. Select "From Connected Devices"
2. No device connected
3. Check error handling

Expected:
✅ Warning shown
✅ Fallback option suggested
✅ No crash
```

### **Test 7.4: Network Error**
```
Steps:
1. Desktop tool tries to upload
2. Simulate network error
3. Check error handling

Expected:
✅ Error message shown
✅ Clear retry option
✅ No crash
```

---

## 🧪 TEST SUITE 8: END-TO-END WORKFLOW

### **Test 8.1: Complete Online Workflow**
```
Steps:
1. Web App: Create case
2. Web App: Get approval
3. Web App: Generate token
4. Web App: Copy token
5. Desktop Tool: Paste token
6. Desktop Tool: Verify token
7. Desktop Tool: Select device
8. Desktop Tool: Extract data
9. Desktop Tool: Upload results
10. Web App: View results

Expected:
✅ All steps complete
✅ No errors
✅ Results visible
✅ Audit trail complete
```

### **Test 8.2: Multiple Cases**
```
Steps:
1. Create 3 different cases
2. Generate tokens for each
3. Test extraction for each
4. Check all results

Expected:
✅ All cases work
✅ All tokens valid
✅ All results stored
✅ No conflicts
```

### **Test 8.3: Concurrent Users**
```
Steps:
1. Open web app in 2 browsers
2. Create cases in both
3. Generate tokens in both
4. Extract in both

Expected:
✅ Both work independently
✅ No data conflicts
✅ Session state separate
```

---

## 📋 TESTING CHECKLIST

### **Before Deployment**
- [ ] All offline tests pass (15/15)
- [ ] Code review complete
- [ ] No console errors
- [ ] All imports working
- [ ] Dependencies installed

### **After Deployment**
- [ ] App loads on Streamlit Cloud
- [ ] Navigation works
- [ ] Cases can be created
- [ ] Tokens can be generated
- [ ] Desktop tool can verify tokens
- [ ] Results can be uploaded
- [ ] All 8 test suites pass

### **Ongoing Monitoring**
- [ ] Check app logs daily
- [ ] Monitor error rates
- [ ] Track user feedback
- [ ] Performance metrics
- [ ] Uptime monitoring

---

## 🚀 TESTING EXECUTION

### **Step 1: Offline Testing** ✅
```bash
python test_offline_suite.py
# Expected: 15/15 tests pass
```

### **Step 2: Deploy to Streamlit Cloud**
```bash
git push origin main
# Streamlit Cloud auto-deploys
```

### **Step 3: Online Testing** ⏳
```
1. Go to https://forensmart.streamlit.app
2. Run through all 8 test suites
3. Document results
4. Fix any issues
```

### **Step 4: Integration Testing** ⏳
```
1. Test web app + desktop tool together
2. Test token generation and verification
3. Test extraction and upload
4. Test error scenarios
```

---

## 📊 TEST RESULTS TEMPLATE

```
Date: [DATE]
Tester: [NAME]
Environment: Streamlit Cloud
App URL: https://forensmart.streamlit.app

TEST SUITE 1: WEB APP FUNCTIONALITY
- Test 1.1: App Loads ✅/❌
- Test 1.2: Navigation Works ✅/❌
- Test 1.3: Session State Persists ✅/❌

TEST SUITE 2: CASE MANAGEMENT
- Test 2.1: Create Case ✅/❌
- Test 2.2: View Cases ✅/❌
- Test 2.3: Case Details ✅/❌

TEST SUITE 3: DEVICE DETECTION
- Test 3.1: Device Selection Options ✅/❌
- Test 3.2: From Your Cases ✅/❌
- Test 3.3: Manual Entry ✅/❌
- Test 3.4: From Connected Devices ✅/❌

TEST SUITE 4: TOKEN GENERATION
- Test 4.1: Generate Token ✅/❌
- Test 4.2: Copy Token ✅/❌
- Test 4.3: Download Token ✅/❌

TEST SUITE 5: EXTRACTION WORKFLOW
- Test 5.1: Module Selection ✅/❌
- Test 5.2: Consent Verification ✅/❌
- Test 5.3: Extraction Progress ✅/❌
- Test 5.4: Results Display ✅/❌

TEST SUITE 6: DESKTOP TOOL INTEGRATION
- Test 6.1: Token from Web App ✅/❌
- Test 6.2: Device Detection ✅/❌
- Test 6.3: Extraction ✅/❌
- Test 6.4: Results Upload ✅/❌

TEST SUITE 7: ERROR HANDLING
- Test 7.1: Invalid Token ✅/❌
- Test 7.2: Expired Token ✅/❌
- Test 7.3: Missing Device ✅/❌
- Test 7.4: Network Error ✅/❌

TEST SUITE 8: END-TO-END WORKFLOW
- Test 8.1: Complete Workflow ✅/❌
- Test 8.2: Multiple Cases ✅/❌
- Test 8.3: Concurrent Users ✅/❌

TOTAL: [X]/[Y] PASSED
SUCCESS RATE: [X]%

ISSUES FOUND:
1. [Issue 1]
2. [Issue 2]

NOTES:
[Any additional notes]
```

---

## ✅ TESTING BENEFITS

- ✅ **Comprehensive Coverage** - 8 test suites, 30+ tests
- ✅ **Real-World Scenarios** - Tests actual user workflows
- ✅ **Error Scenarios** - Tests error handling
- ✅ **Integration Testing** - Tests web app + desktop tool
- ✅ **Performance Testing** - Tests multiple users
- ✅ **Documentation** - Clear test procedures

---

## 🚀 STATUS

**Offline Testing**: ✅ **READY** (15 tests)

**Online Testing**: 📋 **PLANNED** (30+ tests)

**Integration Testing**: 📋 **PLANNED**

**Deployment**: ⏳ **AFTER TESTING**

---

**Status**: 🚀 **READY FOR ONLINE DEPLOYMENT & TESTING**
