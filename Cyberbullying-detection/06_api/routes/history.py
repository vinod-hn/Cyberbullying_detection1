"""
Dashboard History API Routes
Handles storage and retrieval of analysis sessions for the View History feature.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import sqlite3
from pathlib import Path

router = APIRouter(prefix="/history", tags=["history"])

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "local.db"


def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_history_table():
    """Initialize the dashboard_history table if it doesn't exist."""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                cyberbullying_count INTEGER DEFAULT 0,
                neutral_count INTEGER DEFAULT 0,
                json_results TEXT,
                summary_stats TEXT,
                severity_distribution TEXT
            )
        """)
        conn.commit()
        print("Dashboard history table initialized")
    except Exception as e:
        print(f"Error initializing history table: {e}")
    finally:
        conn.close()


# Initialize table on module load
init_history_table()


class HistoryEntry(BaseModel):
    """Model for history entry response."""
    id: int
    filename: str
    uploaded_at: str
    message_count: int
    cyberbullying_count: int
    neutral_count: int
    severity_distribution: Optional[dict] = None


class HistoryDetail(HistoryEntry):
    """Model for detailed history entry with full data."""
    json_results: Optional[List[dict]] = None
    summary_stats: Optional[dict] = None


class SaveHistoryRequest(BaseModel):
    """Request model for saving history."""
    filename: str
    messages: List[dict]
    summary_stats: Optional[dict] = None


@router.get("", response_model=List[HistoryEntry])
async def get_history(limit: int = 10):
    """
    Get list of previous analysis sessions.
    Returns the most recent sessions with summary info.
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            SELECT id, filename, uploaded_at, message_count, 
                   cyberbullying_count, neutral_count, severity_distribution
            FROM dashboard_history
            ORDER BY uploaded_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        history = []
        
        for row in rows:
            entry = {
                "id": row["id"],
                "filename": row["filename"],
                "uploaded_at": row["uploaded_at"],
                "message_count": row["message_count"],
                "cyberbullying_count": row["cyberbullying_count"],
                "neutral_count": row["neutral_count"],
                "severity_distribution": None
            }
            
            if row["severity_distribution"]:
                try:
                    entry["severity_distribution"] = json.loads(row["severity_distribution"])
                except:
                    pass
            
            history.append(entry)
        
        return history
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")
    finally:
        conn.close()


@router.get("/{history_id}", response_model=HistoryDetail)
async def get_history_detail(history_id: int):
    """
    Get full details of a specific history entry.
    Includes all messages and results for restoring the dashboard.
    """
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM dashboard_history WHERE id = ?
        """, (history_id,))
        
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        detail = {
            "id": row["id"],
            "filename": row["filename"],
            "uploaded_at": row["uploaded_at"],
            "message_count": row["message_count"],
            "cyberbullying_count": row["cyberbullying_count"],
            "neutral_count": row["neutral_count"],
            "json_results": None,
            "summary_stats": None,
            "severity_distribution": None
        }
        
        if row["json_results"]:
            try:
                detail["json_results"] = json.loads(row["json_results"])
            except:
                pass
        
        if row["summary_stats"]:
            try:
                detail["summary_stats"] = json.loads(row["summary_stats"])
            except:
                pass
        
        if row["severity_distribution"]:
            try:
                detail["severity_distribution"] = json.loads(row["severity_distribution"])
            except:
                pass
        
        return detail
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history detail: {str(e)}")
    finally:
        conn.close()


@router.post("", response_model=HistoryEntry)
async def save_history(request: SaveHistoryRequest):
    """
    Save a new analysis session to history.
    Called automatically after CSV upload and analysis.
    """
    conn = get_db_connection()
    try:
        messages = request.messages
        
        # Calculate counts
        message_count = len(messages)
        cyberbullying_count = sum(1 for m in messages if m.get("is_cyberbullying", False) or 
                                  m.get("severity", "").lower() not in ["neutral", "safe", "low", ""])
        neutral_count = message_count - cyberbullying_count
        
        # Calculate severity distribution
        severity_dist = {}
        for m in messages:
            sev = (m.get("score") or m.get("prediction") or m.get("severity") or "unknown").lower()
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
        
        # Build summary stats
        summary_stats = request.summary_stats or {
            "total": message_count,
            "cyberbullying": cyberbullying_count,
            "neutral": neutral_count,
            "severity_distribution": severity_dist
        }
        
        # Insert into database
        cursor = conn.execute("""
            INSERT INTO dashboard_history 
            (filename, message_count, cyberbullying_count, neutral_count, 
             json_results, summary_stats, severity_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request.filename,
            message_count,
            cyberbullying_count,
            neutral_count,
            json.dumps(messages),
            json.dumps(summary_stats),
            json.dumps(severity_dist)
        ))
        
        conn.commit()
        history_id = cursor.lastrowid
        
        # Fetch and return the created entry
        cursor = conn.execute("""
            SELECT id, filename, uploaded_at, message_count, 
                   cyberbullying_count, neutral_count, severity_distribution
            FROM dashboard_history WHERE id = ?
        """, (history_id,))
        
        row = cursor.fetchone()
        
        return {
            "id": row["id"],
            "filename": row["filename"],
            "uploaded_at": row["uploaded_at"],
            "message_count": row["message_count"],
            "cyberbullying_count": row["cyberbullying_count"],
            "neutral_count": row["neutral_count"],
            "severity_distribution": severity_dist
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {str(e)}")
    finally:
        conn.close()


@router.delete("/{history_id}")
async def delete_history(history_id: int):
    """Delete a history entry."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("DELETE FROM dashboard_history WHERE id = ?", (history_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        return {"message": "History entry deleted", "id": history_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")
    finally:
        conn.close()
