-- Create approval_links table
CREATE TABLE IF NOT EXISTS approval_links (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    token VARCHAR(255) UNIQUE NOT NULL,
    nominee_email VARCHAR(255) NOT NULL,
    consent_level VARCHAR(50) NOT NULL,
    approval_method VARCHAR(50),
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Create consent_approvals table
CREATE TABLE IF NOT EXISTS consent_approvals (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    nominee_email VARCHAR(255) NOT NULL,
    approval_link_id INTEGER REFERENCES approval_links(id),
    consent_level VARCHAR(50) NOT NULL,
    approval_method VARCHAR(50),
    approved_at TIMESTAMP,
    approved_by VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    pin_hash VARCHAR(255),
    pattern_hash VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create approval_history table
CREATE TABLE IF NOT EXISTS approval_history (
    id SERIAL PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL,
    approval_link_id INTEGER REFERENCES approval_links(id),
    approval_id INTEGER REFERENCES consent_approvals(id),
    action VARCHAR(100) NOT NULL,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_email VARCHAR(255),
    ip_address VARCHAR(50)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_approval_links_case_id ON approval_links(case_id);
CREATE INDEX IF NOT EXISTS idx_approval_links_token ON approval_links(token);
CREATE INDEX IF NOT EXISTS idx_consent_approvals_case_id ON consent_approvals(case_id);
CREATE INDEX IF NOT EXISTS idx_approval_history_case_id ON approval_history(case_id);
CREATE INDEX IF NOT EXISTS idx_approval_history_timestamp ON approval_history(timestamp);

-- Grant permissions to forensmart_user
GRANT ALL PRIVILEGES ON approval_links TO forensmart_user;
GRANT ALL PRIVILEGES ON consent_approvals TO forensmart_user;
GRANT ALL PRIVILEGES ON approval_history TO forensmart_user;
GRANT ALL PRIVILEGES ON approval_links_id_seq TO forensmart_user;
GRANT ALL PRIVILEGES ON consent_approvals_id_seq TO forensmart_user;
GRANT ALL PRIVILEGES ON approval_history_id_seq TO forensmart_user;
