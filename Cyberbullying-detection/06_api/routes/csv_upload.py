"""
CSV Upload endpoint for chat analysis.
POST /upload/csv
"""

import io
import csv
import uuid
import time
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from ..models_loader import get_detector
    from ..schemas import PredictionResponse
except ImportError:
    from models_loader import get_detector
    from schemas import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Rate limiting state (simple in-memory)
_rate_limit_store = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 10  # max uploads per window per IP


class CSVAnalysisResult(BaseModel):
    """Result for a single analyzed message."""
    row_id: int
    original_text: str
    prediction: str
    confidence: float
    is_cyberbullying: bool
    severity: str
    model_used: str
    probabilities: Optional[dict] = None


class CSVUploadResponse(BaseModel):
    """Response for CSV upload and analysis."""
    success: bool
    message: str
    upload_id: str
    total_rows: int
    analyzed_rows: int
    cyberbullying_count: int
    not_cyberbullying_count: int
    severity_distribution: dict
    processing_time_ms: float
    results: List[CSVAnalysisResult]
    preview: List[dict] = Field(default_factory=list)


def check_rate_limit(ip_address: str) -> bool:
    """Check if request is within rate limit."""
    now = time.time()
    
    # Clean old entries
    _rate_limit_store[ip_address] = [
        t for t in _rate_limit_store.get(ip_address, [])
        if now - t < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(_rate_limit_store.get(ip_address, [])) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    
    # Add current request
    if ip_address not in _rate_limit_store:
        _rate_limit_store[ip_address] = []
    _rate_limit_store[ip_address].append(now)
    
    return True


def determine_severity(confidence: float, is_cyberbullying: bool) -> str:
    """Determine severity level based on confidence and prediction."""
    if not is_cyberbullying:
        return "low"
    
    if confidence >= 0.9:
        return "critical"
    elif confidence >= 0.75:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


# Columns to EXCLUDE from text detection (metadata columns)
METADATA_COLUMNS = [
    'name', 'username', 'user', 'sender', 'from', 'author',
    'time', 'timestamp', 'datetime', 'date', 'created_at', 'sent_at',
    'id', 'chat_id', 'message_id', 'student_id', 'user_id',
    'platform', 'source', 'channel', 'group',
    'phone', 'number', 'email'
]

import re

# Regex patterns for WhatsApp-style message formats
WHATSAPP_PATTERNS = [
    # Pattern: DD/MM/YYYY, HH:MM am/pm - Name: Message
    re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\s*-\s*[^:]+:\s*(.+)$'),
    # Pattern: [DD/MM/YYYY, HH:MM:SS] Name: Message
    re.compile(r'^\[\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\]\s*[^:]+:\s*(.+)$'),
    # Pattern: YYYY-MM-DD HH:MM - Name: Message
    re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s*-\s*[^:]+:\s*(.+)$'),
    # Pattern: Name: Message (simple format with name prefix)
    re.compile(r'^[A-Za-z][A-Za-z0-9\s_\-\.]+:\s+(.+)$'),
]


def extract_message_content(text: str) -> str:
    """
    Extract only the actual message content from formatted chat messages.
    Removes date, time, and sender name prefixes (WhatsApp, Telegram, etc.)
    
    Examples:
    - "15/01/2026, 9:49 am - Reshma Begum Akka: Enjoy your festival" -> "Enjoy your festival"
    - "[15/01/2026, 9:49:00 AM] John: Hello" -> "Hello"
    - "User123: Hi there" -> "Hi there"
    """
    if not text:
        return text
    
    text = text.strip()
    
    # Try each pattern
    for pattern in WHATSAPP_PATTERNS:
        match = pattern.match(text)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    
    # If no pattern matched, return original text
    return text

def validate_csv_structure(headers: List[str]) -> tuple:
    """
    Validate CSV structure and find the text/message column.
    Excludes metadata columns like name, time, id, etc.
    Returns (is_valid, text_column_name, error_message)
    """
    # Priority order for text column names
    text_column_names = [
        'message', 'text', 'content', 'chat', 'comment', 
        'msg', 'body', 'post', 'sentence', 'input', 'tweet'
    ]
    
    headers_lower = [h.lower().strip() for h in headers]
    
    # First try to find an exact match for known text columns
    for col_name in text_column_names:
        if col_name in headers_lower:
            idx = headers_lower.index(col_name)
            return True, headers[idx], None
    
    # If no known column found, find the first column that's NOT metadata
    for i, h in enumerate(headers_lower):
        is_metadata = any(meta in h for meta in METADATA_COLUMNS)
        if not is_metadata:
            return True, headers[i], f"Using column '{headers[i]}' as text column (auto-detected)"
    
    # Last resort: use the column with longest average content
    if len(headers) > 0:
        return True, headers[0], f"Using first column '{headers[0]}' as text column"
    
    return False, None, "CSV must have at least one column"


@router.post("/upload/csv", response_model=CSVUploadResponse)
async def upload_csv(
    request: Request,
    file: UploadFile = File(..., description="CSV or TXT file containing chat messages"),
    model_type: str = Form(default="bert", description="Model to use for analysis"),
    text_column: Optional[str] = Form(default=None, description="Name of the column containing text"),
    max_rows: Optional[int] = Form(default=1000, description="Maximum rows to process")
):
    """
    Upload a CSV or TXT file for chat message analysis.
    
    For CSV: Should contain a column with chat messages/text to analyze.
    Common column names detected: message, text, content, chat, comment
    
    For TXT: One message per line.
    
    Returns analysis results with predictions, severity, and confidence scores.
    """
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 10 uploads per minute."
        )
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    filename_lower = file.filename.lower()
    is_txt = filename_lower.endswith('.txt')
    is_csv = filename_lower.endswith('.csv')
    
    if not is_txt and not is_csv:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a CSV or TXT file."
        )
    
    # Check file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    start_time = time.time()
    upload_id = str(uuid.uuid4())
    
    try:
        decoded = contents.decode('utf-8')
        
        if is_txt:
            # Parse TXT file (one message per line)
            # Each line may contain WhatsApp-style format: date, time - name: message
            lines = [line.strip() for line in decoded.split('\n') if line.strip()]
            # Store raw line but message will be extracted during analysis
            rows = [{"message": line} for line in lines]
            headers = ["message"]
            target_column = "message"
            total_rows = len(rows)
        else:
            # Parse CSV
            reader = csv.DictReader(io.StringIO(decoded))
            
            if not reader.fieldnames:
                raise HTTPException(status_code=400, detail="CSV file is empty or has no headers")
            
            headers = list(reader.fieldnames)
            
            # Validate and find text column
            if text_column and text_column in headers:
                target_column = text_column
            else:
                is_valid, target_column, warning = validate_csv_structure(headers)
                if not is_valid:
                    raise HTTPException(status_code=400, detail=warning)
                if warning:
                    logger.info(warning)
            
            # Read all rows
            rows = list(reader)
            total_rows = len(rows)
        
        if total_rows == 0:
            raise HTTPException(status_code=400, detail="CSV file contains no data rows")
        
        # Limit rows
        rows_to_process = rows[:max_rows]
        
        # Get detector
        try:
            detector = get_detector(model_type)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model '{model_type}'. Available: bert, mbert, indicbert, baseline"
            )
        
        # Analyze messages
        results = []
        cyberbullying_count = 0
        not_cyberbullying_count = 0
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for idx, row in enumerate(rows_to_process):
            raw_text = row.get(target_column, "").strip()
            
            if not raw_text:
                continue
            
            # Extract only message content (remove date, time, sender name)
            text = extract_message_content(raw_text)
            
            if not text:
                continue
            
            # Log if extraction was applied
            if text != raw_text:
                logger.debug(f"Extracted message: '{raw_text[:50]}...' -> '{text[:50]}...'")
            
            try:
                prediction = detector.predict(text)
                
                is_cb = prediction.get("is_cyberbullying", False)
                confidence = prediction.get("confidence", 0.0)
                severity = determine_severity(confidence, is_cb)
                
                if is_cb:
                    cyberbullying_count += 1
                else:
                    not_cyberbullying_count += 1
                
                severity_counts[severity] += 1
                
                result = CSVAnalysisResult(
                    row_id=idx + 1,
                    original_text=text[:500],  # Truncate long texts
                    prediction=prediction.get("prediction", "Unknown"),
                    confidence=round(confidence, 4),
                    is_cyberbullying=is_cb,
                    severity=severity,
                    model_used=model_type,
                    probabilities=prediction.get("probabilities")
                )
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to analyze row {idx + 1}: {e}")
                continue
        
        # Generate preview (first 5 rows with all columns)
        preview = []
        for row in rows[:5]:
            preview_row = {k: str(v)[:100] for k, v in row.items()}
            preview.append(preview_row)
        
        processing_time = (time.time() - start_time) * 1000
        
        return CSVUploadResponse(
            success=True,
            message=f"Successfully analyzed {len(results)} messages",
            upload_id=upload_id,
            total_rows=total_rows,
            analyzed_rows=len(results),
            cyberbullying_count=cyberbullying_count,
            not_cyberbullying_count=not_cyberbullying_count,
            severity_distribution=severity_counts,
            processing_time_ms=round(processing_time, 2),
            results=results,
            preview=preview
        )
        
    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid file encoding. Please use UTF-8 encoded CSV."
        )
    except csv.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parsing error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"CSV upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing CSV: {str(e)}"
        )


@router.get("/upload/validate")
async def validate_csv_preview(
    request: Request,
    file: UploadFile = File(..., description="CSV file to validate")
):
    """
    Validate a CSV file without processing.
    Returns headers, row count, and detected text column.
    """
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        contents = await file.read()
        decoded = contents.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded))
        
        headers = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)
        
        is_valid, text_column, message = validate_csv_structure(headers)
        
        # Preview first 5 rows
        preview = []
        for row in rows[:5]:
            preview.append({k: str(v)[:100] for k, v in row.items()})
        
        return {
            "valid": is_valid,
            "headers": headers,
            "detected_text_column": text_column,
            "total_rows": len(rows),
            "preview": preview,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
