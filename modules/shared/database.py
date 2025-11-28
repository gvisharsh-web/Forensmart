"""
DATABASE MODULE - Database operations and management

Provides:
- Database connection management
- CRUD operations
- Query execution
- Transaction management
- Data persistence
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE CLASS
# ============================================================================

class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, db_type: str = "sqlite"):
        self.db_type = db_type
        self.connection = None
        self.data_store = {}  # In-memory store for demo
        self.transaction_history = []
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect(self, connection_string: str = None) -> bool:
        """Connect to database"""
        try:
            logger.info(f"Connecting to {self.db_type} database")
            self.connection = True
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from database"""
        try:
            logger.info("Disconnecting from database")
            self.connection = None
            return True
        except Exception as e:
            logger.error(f"Disconnection failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connection is not None
    
    # ========================================================================
    # CRUD OPERATIONS
    # ========================================================================
    
    def create(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create record"""
        try:
            if table not in self.data_store:
                self.data_store[table] = []
            
            record = {
                'id': len(self.data_store[table]) + 1,
                **data,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            self.data_store[table].append(record)
            logger.info(f"Created record in {table}")
            
            return record
        except Exception as e:
            logger.error(f"Create failed: {e}")
            return {}
    
    def read(self, table: str, record_id: int = None) -> List[Dict[str, Any]]:
        """Read records"""
        try:
            if table not in self.data_store:
                return []
            
            if record_id:
                records = [r for r in self.data_store[table] if r.get('id') == record_id]
            else:
                records = self.data_store[table]
            
            logger.info(f"Read {len(records)} records from {table}")
            return records
        except Exception as e:
            logger.error(f"Read failed: {e}")
            return []
    
    def update(self, table: str, record_id: int, data: Dict[str, Any]) -> bool:
        """Update record"""
        try:
            if table not in self.data_store:
                return False
            
            for record in self.data_store[table]:
                if record.get('id') == record_id:
                    record.update(data)
                    record['updated_at'] = datetime.now().isoformat()
                    logger.info(f"Updated record {record_id} in {table}")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def delete(self, table: str, record_id: int) -> bool:
        """Delete record"""
        try:
            if table not in self.data_store:
                return False
            
            self.data_store[table] = [r for r in self.data_store[table] if r.get('id') != record_id]
            logger.info(f"Deleted record {record_id} from {table}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    def query(self, table: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query records with filters"""
        try:
            if table not in self.data_store:
                return []
            
            records = self.data_store[table]
            
            if filters:
                for key, value in filters.items():
                    records = [r for r in records if r.get(key) == value]
            
            logger.info(f"Query returned {len(records)} records")
            return records
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def aggregate(self, table: str, operation: str, field: str) -> Any:
        """Aggregate operation"""
        try:
            if table not in self.data_store:
                return None
            
            records = self.data_store[table]
            values = [r.get(field) for r in records if field in r]
            
            if operation == 'count':
                return len(values)
            elif operation == 'sum':
                return sum(v for v in values if isinstance(v, (int, float)))
            elif operation == 'avg':
                if values:
                    return sum(v for v in values if isinstance(v, (int, float))) / len(values)
            elif operation == 'max':
                return max(values) if values else None
            elif operation == 'min':
                return min(values) if values else None
            
            return None
        except Exception as e:
            logger.error(f"Aggregate failed: {e}")
            return None
    
    # ========================================================================
    # TRANSACTION MANAGEMENT
    # ========================================================================
    
    def begin_transaction(self) -> str:
        """Begin transaction"""
        transaction_id = f"txn_{datetime.now().timestamp()}"
        self.transaction_history.append({
            'id': transaction_id,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Transaction {transaction_id} started")
        return transaction_id
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit transaction"""
        try:
            for txn in self.transaction_history:
                if txn['id'] == transaction_id:
                    txn['status'] = 'committed'
                    logger.info(f"Transaction {transaction_id} committed")
                    return True
            return False
        except Exception as e:
            logger.error(f"Commit failed: {e}")
            return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback transaction"""
        try:
            for txn in self.transaction_history:
                if txn['id'] == transaction_id:
                    txn['status'] = 'rolled_back'
                    logger.info(f"Transaction {transaction_id} rolled back")
                    return True
            return False
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'tables': len(self.data_store),
            'total_records': sum(len(records) for records in self.data_store.values()),
            'transactions': len(self.transaction_history),
            'timestamp': datetime.now().isoformat()
        }
        
        for table, records in self.data_store.items():
            stats[f'{table}_count'] = len(records)
        
        return stats
    
    def export_data(self, table: str = None) -> str:
        """Export data to JSON"""
        try:
            if table:
                data = self.data_store.get(table, [])
            else:
                data = self.data_store
            
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ""
    
    def clear_data(self, table: str = None) -> bool:
        """Clear data"""
        try:
            if table:
                self.data_store[table] = []
            else:
                self.data_store = {}
            
            logger.info("Data cleared")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def create_database_manager(db_type: str = "sqlite") -> DatabaseManager:
    """Factory function to create database manager"""
    return DatabaseManager(db_type)
