# 🚀 Supabase Setup Guide - ForenSmart Hybrid Approval System

## **Quick Start (5 minutes)**

### **Step 1: Create Free Supabase Account**
1. Go to https://supabase.com
2. Click "Start your project"
3. Sign up with GitHub or Google
4. Create new project (select FREE tier)
5. Wait for project to initialize (2-3 minutes)

### **Step 2: Create Approvals Table**
1. In Supabase dashboard, go to **SQL Editor**
2. Click **New Query**
3. Copy and paste this SQL:

```sql
CREATE TABLE approvals (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  case_id TEXT UNIQUE NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('approved', 'denied')),
  nominee_name TEXT DEFAULT '',
  timestamp TIMESTAMP DEFAULT NOW(),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX idx_approvals_case_id ON approvals(case_id);
CREATE INDEX idx_approvals_decision ON approvals(decision);
```

4. Click **Run**
5. You should see "Success" message

### **Step 3: Get API Credentials**
1. Go to **Settings** → **API**
2. Copy these values:
   - **Project URL** (e.g., `https://your-project.supabase.co`)
   - **anon public** key (under "Project API keys")
3. Save these for next step

### **Step 4: Configure ForenSmart**

#### **Option A: Local Testing (Development)**
Create `.streamlit/secrets.toml`:
```toml
supabase_url = "https://your-project.supabase.co"
supabase_key = "your-anon-public-key"
```

#### **Option B: Streamlit Cloud (Production)**
1. Go to your Streamlit app settings
2. Click **Secrets**
3. Add:
```toml
supabase_url = "https://your-project.supabase.co"
supabase_key = "your-anon-public-key"
```

### **Step 5: Install Supabase Library**
```bash
pip install supabase
```

### **Step 6: Test Connection**
Run this in Python:
```python
from modules.approval.supabase_client import get_supabase_client

client = get_supabase_client()
health = client.health_check()
print(health)
```

Expected output:
```
{
    'status': 'online',
    'available': True,
    'message': 'Supabase connection healthy'
}
```

---

## **How It Works**

### **Hybrid Approval Flow**

```
User approves extraction
    ↓
HybridApprovalManager.mark_approved()
    ├─ Save to Supabase (online)
    └─ Save to file (offline)
    ↓
Orchestrator checks approval
    ├─ Try Supabase first (online)
    ├─ Fall back to file (offline)
    └─ Return approval status
    ↓
Extraction proceeds if approved
```

### **Local Testing (Offline)**
- No Supabase needed
- Uses file-based approvals
- Perfect for development

### **Cloud Deployment (Online)**
- Uses Supabase
- File system not available
- Seamless fallback if Supabase down

---

## **API Reference**

### **HybridApprovalManager**

```python
from modules.approval.approval_manager_hybrid import get_hybrid_approval_manager

manager = get_hybrid_approval_manager()

# Check approval
result = manager.check_approval_with_fallback(case_id)
# Returns: {'approved': bool, 'status': str, 'source': str, ...}

# Mark as approved
result = manager.mark_approved(case_id, nominee_name="John Doe")
# Returns: {'success': bool, 'online_saved': bool, 'offline_saved': bool}

# Mark as denied
result = manager.mark_denied(case_id, nominee_name="John Doe")
# Returns: {'success': bool, 'online_saved': bool, 'offline_saved': bool}

# Get status
status = manager.get_approval_status(case_id)
# Returns: {'case_id': str, 'is_approved': bool, 'status': str, ...}

# Health check
health = manager.health_check()
# Returns: {'system': str, 'status': str, 'online': {...}, 'offline': {...}}
```

### **SupabaseApprovalClient**

```python
from modules.approval.supabase_client import get_supabase_client

client = get_supabase_client()

# Get approval
approval = client.get_approval(case_id)
# Returns: {'case_id': str, 'decision': str, ...} or None

# Save approval
success = client.save_approval(case_id, 'approved', 'John Doe')
# Returns: bool

# Delete approval
success = client.delete_approval(case_id)
# Returns: bool

# List approvals
approvals = client.list_approvals(limit=100)
# Returns: list of approvals or None

# Health check
health = client.health_check()
# Returns: {'status': str, 'available': bool, 'message': str}
```

---

## **Troubleshooting**

### **Issue: "Supabase credentials not configured"**
**Solution:** Add credentials to `.streamlit/secrets.toml` or Streamlit Cloud Secrets

### **Issue: "Supabase library not installed"**
**Solution:** Run `pip install supabase`

### **Issue: "Connection timeout"**
**Solution:** 
- Check internet connection
- Verify Supabase URL is correct
- Check if Supabase project is active

### **Issue: "Table not found"**
**Solution:** Run the SQL script in Supabase SQL Editor to create table

### **Issue: "Permission denied"**
**Solution:** 
- Check API key is correct
- Verify table permissions in Supabase
- Check row-level security policies

### **Issue: "Approval not saving"**
**Solution:**
- Check logs for error messages
- Verify Supabase connection health
- Check file permissions for offline fallback

---

## **Testing Checklist**

- [ ] Supabase account created
- [ ] Approvals table created
- [ ] API credentials copied
- [ ] `.streamlit/secrets.toml` configured
- [ ] Supabase library installed
- [ ] Connection health check passes
- [ ] Local testing works (file-based)
- [ ] Supabase testing works (online)
- [ ] Fallback works (Supabase → file)
- [ ] Extraction works with approval

---

## **Security Notes**

✅ **Best Practices:**
- Never commit `.streamlit/secrets.toml` to Git
- Use Streamlit Cloud Secrets for production
- Rotate API keys regularly
- Use Row-Level Security (RLS) for sensitive data
- Enable SSL/TLS for all connections

⚠️ **Warnings:**
- Don't share API keys publicly
- Don't use service role key in frontend
- Always validate approval data
- Log all approval changes for audit

---

## **Support**

**For Supabase issues:**
- Docs: https://supabase.com/docs
- Community: https://discord.supabase.io

**For ForenSmart issues:**
- Check logs: `app_logs.txt`
- Check approval files: `audit/approvals/`
- Run health check: `manager.health_check()`

---

## **Next Steps**

1. ✅ Set up Supabase account
2. ✅ Create approvals table
3. ✅ Configure ForenSmart
4. ✅ Test locally
5. ✅ Deploy to Streamlit Cloud
6. ✅ Monitor in production

**You're all set! 🎉**
