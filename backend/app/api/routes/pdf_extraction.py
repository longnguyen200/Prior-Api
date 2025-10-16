import base64
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    PDFExtractedData,
    PDFExtractionRequest,
    PDFExtractionResponse,
)

router = APIRouter(prefix="/pdf-extraction", tags=["pdf-extraction"])
logger = logging.getLogger(__name__)


def read_config(file_path: str) -> dict[str, Any]:
    """Read extraction config from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def pdf_to_images_base64(pdf_bytes: bytes, dpi: int = 200) -> list[str]:
    """
    Convert PDF pages to base64 encoded images.
    
    Args:
        pdf_bytes: PDF file content as bytes
        dpi: Resolution for rendering (default 200)
    
    Returns:
        List of base64 encoded images (one per page)
    """
    images_base64 = []
    
    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Convert each page to image
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Render page to image with zoom factor for higher resolution
        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # Encode to base64
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        images_base64.append(base64_image)
    
    pdf_document.close()
    return images_base64


def extract_with_gemini(
    images_base64: list[str], 
    config_json: dict[str, Any],
    api_key: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    """
    Extract data from PDF images using Gemini API.
    
    Args:
        images_base64: List of base64 encoded images
        config_json: Configuration defining fields to extract
        api_key: Gemini API key
        model_name: Model name to use
        base_url: Base URL for API
    
    Returns:
        Extracted data as dictionary
    """
    # Create detailed prompt for medical records extraction
    ocr_prompt = (
        "You are an expert medical document data extraction assistant. "
        "Carefully analyze all pages of this medical document and extract structured data.\n\n"
        "INSTRUCTIONS:\n"
        "1. Extract ALL available information from the document\n"
        "2. Follow the JSON template structure provided below\n"
        "3. If a field is not present in the document, use null for that field\n"
        "4. For array fields (like medications, allergies), include all items found\n"
        "5. Preserve exact medical terminology, codes (ICD, CPT), and formatting\n"
        "6. Include all dates in their original format\n"
        "7. Be thorough - extract every detail visible in the document\n"
        "8. Return ONLY valid JSON, no additional text or explanations\n\n"
        "JSON TEMPLATE:\n"
        + json.dumps(config_json, ensure_ascii=False, indent=2)
        + "\n\nExtract the data now:"
    )
    
    # Initialize OpenAI client with Gemini endpoint
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Prepare messages with all pages
    content: list[dict[str, Any]] = [{"type": "text", "text": ocr_prompt}]
    
    # Add all page images
    for img_base64 in images_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
        })
    
    # Call API with increased token limit for detailed medical records
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,  # Lower temperature for more consistent extraction
        top_p=0.95,
        max_tokens=8192,  # Increased for detailed medical documents
    )
    
    # Parse response
    response_text = response.choices[0].message.content
    if not response_text:
        raise ValueError("Empty response from API")
    
    # Try to parse JSON
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    return json.loads(response_text)


@router.post("/extract", response_model=PDFExtractionResponse)
async def extract_pdf(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(..., description="PDF file to extract"),
    save_to_db: bool = False,
) -> Any:
    """
    Extract structured data from PDF using Gemini Vision API.
    
    - **file**: PDF file to process (multipart/form-data)
    - **save_to_db**: If True, save extracted data to database
    
    Returns extracted data as JSON.
    """
    # Validate API key
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured in environment",
        )
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )
    
    try:
        # Read PDF file
        pdf_bytes = await file.read()
        
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        # Read extraction config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        config_json = read_config(str(config_path))
        
        # Convert PDF pages to images
        logger.info(f"Converting PDF to images: {file.filename}")
        images_base64 = pdf_to_images_base64(pdf_bytes)
        logger.info(f"Converted {len(images_base64)} pages")
        
        # Extract data using Gemini
        logger.info("Calling Gemini API for extraction")
        extracted_data = extract_with_gemini(
            images_base64=images_base64,
            config_json=config_json,
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME,
            base_url=settings.GEMINI_BASE_URL,
        )
        
        # Save to database if requested
        saved_id = None
        if save_to_db:
            pdf_data = PDFExtractedData(
                filename=file.filename,
                extracted_data=json.dumps(extracted_data, ensure_ascii=False),
                owner_id=current_user.id,
            )
            session.add(pdf_data)
            session.commit()
            session.refresh(pdf_data)
            saved_id = str(pdf_data.id)
            logger.info(f"Saved to database with ID: {pdf_data.id}")
        
        return PDFExtractionResponse(
            success=True,
            extracted_data=extracted_data,
            filename=file.filename,
            id=saved_id,
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse API response as JSON: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        return PDFExtractionResponse(
            success=False,
            error=str(e),
            filename=file.filename or "unknown",
        )


@router.get("/history")
def get_extraction_history(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Get extraction history for current user.
    """
    from sqlmodel import select
    
    statement = (
        select(PDFExtractedData)
        .where(PDFExtractedData.owner_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(PDFExtractedData.created_at.desc())  # type: ignore
    )
    
    results = session.exec(statement).all()
    
    return {
        "data": [
            {
                "id": str(item.id),
                "filename": item.filename,
                "extracted_data": json.loads(item.extracted_data),
                "created_at": item.created_at,
            }
            for item in results
        ],
        "count": len(results),
    }


@router.get("/config")
def get_extraction_config() -> dict[str, Any]:
    """
    Get current extraction configuration template.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    return read_config(str(config_path))


@router.get("/{pdf_id}")
def get_pdf_extraction_detail(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    pdf_id: str,
) -> Any:
    """
    Get detailed PDF extraction data by ID.
    """
    from sqlmodel import select
    
    try:
        pdf_uuid = UUID(pdf_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF ID format")
    
    statement = (
        select(PDFExtractedData)
        .where(PDFExtractedData.id == pdf_uuid)
        .where(PDFExtractedData.owner_id == current_user.id)
    )
    
    pdf_data = session.exec(statement).first()
    
    if not pdf_data:
        raise HTTPException(status_code=404, detail="PDF extraction data not found")
    
    return {
        "id": str(pdf_data.id),
        "filename": pdf_data.filename,
        "extracted_data": json.loads(pdf_data.extracted_data),
        "created_at": pdf_data.created_at,
    }


@router.delete("/{pdf_id}")
def delete_pdf_extraction(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    pdf_id: str,
) -> dict[str, str]:
    """
    Delete PDF extraction data by ID.
    """
    from sqlmodel import select
    
    try:
        pdf_uuid = UUID(pdf_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF ID format")
    
    statement = (
        select(PDFExtractedData)
        .where(PDFExtractedData.id == pdf_uuid)
        .where(PDFExtractedData.owner_id == current_user.id)
    )
    
    pdf_data = session.exec(statement).first()
    
    if not pdf_data:
        raise HTTPException(status_code=404, detail="PDF extraction data not found")
    
    session.delete(pdf_data)
    session.commit()
    
    return {"message": "PDF extraction data deleted successfully"}
