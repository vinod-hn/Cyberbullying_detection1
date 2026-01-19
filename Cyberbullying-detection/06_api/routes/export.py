"""
Export endpoint for downloading reports.
GET /export_reports
"""

from fastapi import APIRouter, HTTPException, Query, Response
from typing import Optional
import logging
import csv
import io
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


def get_predictions_from_db(export_type: str = "all"):
    """Get predictions from database for export."""
    try:
        from ..db_helper import get_db_context, get_prediction_repository
        
        ctx = get_db_context()
        if ctx is None:
            return []
            
        with ctx as db:
            pred_repo = get_prediction_repository(db)
            if not pred_repo:
                return []
            
            # Get all predictions
            predictions = pred_repo.get_recent(limit=1000)
            
            if export_type == "messages":
                return predictions
            elif export_type == "alerts":
                # Filter to only cyberbullying/high severity
                return [p for p in predictions if p.get("is_cyberbullying") or 
                       p.get("severity", "").lower() in ["high", "critical", "threat", "harassment"]]
            else:
                return predictions
                
    except Exception as e:
        logger.warning(f"Failed to get predictions from database: {e}")
        return []


@router.get("/export_reports")
async def export_reports(
    type: Optional[str] = Query("all", description="Export type: all, messages, alerts")
):
    """
    Export predictions as CSV file.
    
    Args:
        type: Type of export (all, messages, alerts)
        
    Returns:
        CSV file content
    """
    try:
        predictions = get_predictions_from_db(type)
        
        # Create CSV in memory
        output = io.StringIO()
        
        if predictions:
            # Get fieldnames from first prediction
            fieldnames = ["id", "text", "prediction", "is_cyberbullying", "confidence", 
                         "severity", "model_type", "created_at", "platform"]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for pred in predictions:
                row = {
                    "id": pred.get("id", ""),
                    "text": pred.get("text", pred.get("original_text", "")),
                    "prediction": pred.get("prediction", pred.get("label", "")),
                    "is_cyberbullying": pred.get("is_cyberbullying", False),
                    "confidence": pred.get("confidence", 0),
                    "severity": pred.get("severity", ""),
                    "model_type": pred.get("model_type", ""),
                    "created_at": pred.get("created_at", ""),
                    "platform": pred.get("platform", "")
                }
                writer.writerow(row)
        else:
            # Empty export with headers
            fieldnames = ["id", "text", "prediction", "is_cyberbullying", "confidence", 
                         "severity", "model_type", "created_at", "platform"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
        
        csv_content = output.getvalue()
        output.close()
        
        # Return as downloadable CSV
        filename = f"cyberbullying_report_{type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/audit_log")
async def get_audit_log(
    limit: Optional[int] = Query(20, description="Number of log entries to return", ge=1, le=100)
):
    """
    Get audit log entries.
    
    Returns recent audit log entries from the database.
    """
    try:
        from ..db_helper import get_db_context
        
        ctx = get_db_context()
        if ctx is None:
            return {"logs": []}
            
        with ctx as db:
            # Try to get audit logs from database (using audit_logs table)
            cursor = db.execute("""
                SELECT id, action, created_at 
                FROM audit_logs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            logs = []
            for row in cursor.fetchall():
                # Format timestamp
                timestamp = row[2]
                if timestamp:
                    try:
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        time_str = dt.strftime("%I:%M %p")
                    except:
                        time_str = timestamp
                else:
                    time_str = ""
                    
                logs.append({
                    "id": row[0],
                    "action": row[1],
                    "icon": "📋",
                    "time": time_str
                })
            
            return {"logs": logs}
            
    except Exception as e:
        logger.warning(f"Failed to get audit log: {e}")
        return {"logs": []}


@router.post("/audit_log")
async def add_audit_log(action: str, icon: str = "📋"):
    """
    Add an entry to the audit log.
    """
    try:
        from ..db_helper import get_db_context
        import uuid
        
        ctx = get_db_context()
        if ctx is None:
            return {"success": False, "message": "Database not available"}
            
        with ctx as db:
            log_id = str(uuid.uuid4())
            db.execute("""
                INSERT INTO audit_logs (log_id, action, resource_type, created_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (log_id, action, icon))
            db.commit()
            
            return {"success": True}
            
    except Exception as e:
        logger.warning(f"Failed to add audit log: {e}")
        return {"success": False, "message": str(e)}
