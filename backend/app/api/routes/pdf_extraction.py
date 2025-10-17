import base64
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from google import genai
from google.genai import types
from PIL import Image
from sqlmodel import func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    PDFBatchExtractionResponse,
    PDFExtractedData,
    PDFExtractionResponse,
)

router = APIRouter(prefix="/pdf-extraction", tags=["pdf-extraction"])
logger = logging.getLogger(__name__)


def read_config(file_path: str) -> dict[str, Any]:
    """Read extraction config from JSON file."""
    with open(file_path, encoding="utf-8") as file:
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


def extract_with_vertex_ai(
    images_base64: list[str],
    config_json: dict[str, Any],
    api_key: str,
    model_name: str,
    is_batch: bool = False,
) -> dict[str, Any]:
    """
    Extract data from PDF images using Vertex AI Gemini.

    Args:
        images_base64: List of base64 encoded images (can be from multiple files)
        config_json: Configuration defining fields to extract
        api_key: Gemini API key
        model_name: Model name to use
        is_batch: If True, indicates images are from multiple files that should be merged

    Returns:
        Extracted data as dictionary
    """
    # Create detailed prompt for medical records extraction
    batch_instruction = ""
    if is_batch:
        batch_instruction = (
            "NOTE: You are analyzing MULTIPLE PDF files for the same patient/entity. "
            "MERGE all information from all files into a single comprehensive record. "
            "If you find duplicate information, keep the most complete/recent version. "
            "If you find conflicting information, include all versions with notes.\n\n"
        )
    
    system_prompt = (
        "You are an expert medical document data extraction assistant. "
        "Carefully analyze all pages of this medical document and extract structured data.\n\n"
        + batch_instruction +
        "INSTRUCTIONS:\n"
        "1. Extract ALL available information from the document(s)\n"
        "2. Follow the JSON template structure provided below\n"
        "3. If a field is not present in the document, use null for that field\n"
        "4. For array fields (like medications, allergies), include all items found across ALL documents\n"
        "5. Preserve exact medical terminology, codes (ICD, CPT), and formatting\n"
        "6. Include all dates in their original format\n"
        "7. Be thorough - extract every detail visible in all documents\n"
        "8. When merging data from multiple files, combine arrays and select the most complete information\n"
        "9. Return ONLY valid JSON, no additional text or explanations\n\n"
        "JSON TEMPLATE:\n" + json.dumps(config_json, ensure_ascii=False, indent=2)
    )

    # Initialize Vertex AI client
    client = genai.Client(api_key=api_key, vertexai=True)

    # Prepare image inputs
    image_inputs = []
    for img_base64 in images_base64:
        # Convert base64 to PIL Image
        img_data = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_data))
        image_inputs.append(image)

    # Prepare configuration with system instruction
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    # Generate content with images
    response = client.models.generate_content(
        model=model_name,
        contents=image_inputs,
        config=config,
    )

    # Parse response
    response_text = response.text
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
    dedup_mode: Literal["skip", "overwrite", "error"] = Query(
        default="skip",
        description="Deduplication mode: 'skip' returns existing, 'overwrite' replaces, 'error' raises error",
    ),
) -> Any:
    """
    Extract structured data from PDF using Gemini Vision API.

    - **file**: PDF file to process (multipart/form-data)
    - **save_to_db**: If True, save extracted data to database
    - **dedup_mode**: How to handle duplicate files:
        - 'skip': Return existing extraction without re-processing
        - 'overwrite': Delete old and create new extraction
        - 'error': Raise error if duplicate found

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

        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        logger.info(f"File hash: {file_hash}")

        # Check for existing extraction if saving to DB
        existing_record = None
        if save_to_db:
            statement = select(PDFExtractedData).where(
                PDFExtractedData.owner_id == current_user.id,
                PDFExtractedData.file_hash == file_hash,
            )
            existing_record = session.exec(statement).first()

            if existing_record:
                logger.info(f"Found existing extraction with ID: {existing_record.id}")

                if dedup_mode == "error":
                    raise HTTPException(
                        status_code=409,
                        detail=f"This file has already been processed. Existing record ID: {existing_record.id}",
                    )
                elif dedup_mode == "skip":
                    # Return existing extraction without re-processing
                    logger.info("Returning existing extraction (skip mode)")
                    return PDFExtractionResponse(
                        success=True,
                        extracted_data=json.loads(existing_record.extracted_data),
                        filename=existing_record.filename,
                        id=str(existing_record.id),
                    )
                elif dedup_mode == "overwrite":
                    # Delete existing record, will create new one below
                    logger.info("Deleting existing extraction (overwrite mode)")
                    session.delete(existing_record)
                    session.commit()
                    existing_record = None

        # Read extraction config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        config_json = read_config(str(config_path))

        # Convert PDF pages to images
        logger.info(f"Converting PDF to images: {file.filename}")
        images_base64 = pdf_to_images_base64(pdf_bytes)
        logger.info(f"Converted {len(images_base64)} pages")

        # Extract data using Vertex AI Gemini
        logger.info("Calling Vertex AI Gemini API for extraction")
        extracted_data = extract_with_vertex_ai(
            images_base64=images_base64,
            config_json=config_json,
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME,
        )

        # Save to database if requested
        saved_id = None
        if save_to_db:
            pdf_data = PDFExtractedData(
                filename=file.filename,
                file_hash=file_hash,
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


@router.post("/extract-batch", response_model=PDFBatchExtractionResponse)
async def extract_pdf_batch(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    files: list[UploadFile] = File(..., description="Multiple PDF files to extract and merge"),
    save_to_db: bool = False,
    dedup_mode: Literal["skip", "overwrite", "error"] = Query(
        default="skip",
        description="Deduplication mode: 'skip' returns existing, 'overwrite' replaces, 'error' raises error",
    ),
) -> Any:
    """
    Extract and merge structured data from multiple PDF files using Gemini Vision API.
    
    All files are processed together and merged into a single extraction result.

    - **files**: List of PDF files to process and merge (multipart/form-data)
    - **save_to_db**: If True, save merged extracted data to database as single record
    - **dedup_mode**: How to handle duplicate file sets:
        - 'skip': Return existing extraction without re-processing
        - 'overwrite': Delete old and create new extraction
        - 'error': Raise error if duplicate found

    Returns merged extracted data from all files as single JSON.
    """
    # Validate API key
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured in environment",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided",
        )

    filenames = []
    all_images_base64 = []
    all_pdf_bytes = []

    try:
        # Process all files and collect images
        for file in files:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported. Invalid file: {file.filename}",
                )

            filenames.append(file.filename)

            # Read PDF file
            pdf_bytes = await file.read()

            if len(pdf_bytes) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Empty PDF file: {file.filename}"
                )

            all_pdf_bytes.append(pdf_bytes)

            # Convert PDF pages to images
            logger.info(f"Converting PDF to images: {file.filename}")
            images_base64 = pdf_to_images_base64(pdf_bytes)
            logger.info(f"Converted {len(images_base64)} pages from {file.filename}")
            
            # Add all images from this file to the combined list
            all_images_base64.extend(images_base64)

        # Calculate combined hash for deduplication
        combined_bytes = b"".join(all_pdf_bytes)
        combined_hash = hashlib.sha256(combined_bytes).hexdigest()
        logger.info(f"Combined hash for {len(files)} files: {combined_hash}")

        # Check for existing extraction if saving to DB
        existing_record = None
        if save_to_db:
            statement = select(PDFExtractedData).where(
                PDFExtractedData.owner_id == current_user.id,
                PDFExtractedData.file_hash == combined_hash,
            )
            existing_record = session.exec(statement).first()

            if existing_record:
                logger.info(
                    f"Found existing extraction with ID: {existing_record.id}"
                )

                if dedup_mode == "error":
                    raise HTTPException(
                        status_code=409,
                        detail=f"This file set has already been processed. Existing record ID: {existing_record.id}",
                    )
                elif dedup_mode == "skip":
                    # Return existing extraction without re-processing
                    logger.info("Returning existing extraction (skip mode)")
                    return PDFBatchExtractionResponse(
                        success=True,
                        extracted_data=json.loads(existing_record.extracted_data),
                        filenames=json.loads(existing_record.filename) if existing_record.filename.startswith("[") else [existing_record.filename],
                        total_files=len(filenames),
                        id=str(existing_record.id),
                    )
                elif dedup_mode == "overwrite":
                    # Delete existing record, will create new one below
                    logger.info("Deleting existing extraction (overwrite mode)")
                    session.delete(existing_record)
                    session.commit()
                    existing_record = None

        # Read extraction config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        config_json = read_config(str(config_path))

        # Extract and merge data using Vertex AI Gemini with ALL images from ALL files
        logger.info(
            f"Calling Vertex AI Gemini API for merged extraction of {len(files)} files "
            f"({len(all_images_base64)} total pages)"
        )
        
        # Call with is_batch=True to merge information from multiple files
        extracted_data = extract_with_vertex_ai(
            images_base64=all_images_base64,
            config_json=config_json,
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME,
            is_batch=True,
        )

        # Save to database if requested
        saved_id = None
        if save_to_db:
            pdf_data = PDFExtractedData(
                filename=json.dumps(filenames, ensure_ascii=False),  # Store as JSON array
                file_hash=combined_hash,
                extracted_data=json.dumps(extracted_data, ensure_ascii=False),
                owner_id=current_user.id,
            )
            session.add(pdf_data)
            session.commit()
            session.refresh(pdf_data)
            saved_id = str(pdf_data.id)
            logger.info(f"Saved merged extraction to database with ID: {pdf_data.id}")

        return PDFBatchExtractionResponse(
            success=True,
            extracted_data=extracted_data,
            filenames=filenames,
            total_files=len(files),
            id=saved_id,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return PDFBatchExtractionResponse(
            success=False,
            error=f"Failed to parse API response as JSON: {str(e)}",
            filenames=filenames,
            total_files=len(files),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting PDFs: {e}")
        return PDFBatchExtractionResponse(
            success=False,
            error=str(e),
            filenames=filenames,
            total_files=len(files),
        )


@router.get("/patient")
def get_patient_extractions(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Get all patient extractions with pagination (no authentication required).
    """
    # Get total count
    count_statement = select(func.count(PDFExtractedData.id))
    total_count = session.exec(count_statement).one()

    # Get paginated results
    statement = (
        select(PDFExtractedData)
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
                "filenames": json.loads(item.filename) if item.filename.startswith("[") else [item.filename],
                "extracted_data": json.loads(item.extracted_data),
                "created_at": item.created_at,
            }
            for item in results
        ],
        "count": len(results),
        "total": total_count,
        "skip": skip,
        "limit": limit,
    }


@router.get("/config")
def get_extraction_config() -> dict[str, Any]:
    """
    Get current extraction configuration template.
    """
    config_path = Path(__file__).parent.parent.parent / "config.json"
    return read_config(str(config_path))


@router.get("/{patientId}")
def get_patient_data(
    *,
    session: SessionDep,
    patientId: str,
) -> Any:
    """
    Get detailed patient data by ID (no authentication required).
    """

    try:
        patient_uuid = UUID(patientId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    statement = select(PDFExtractedData).where(PDFExtractedData.id == patient_uuid)

    patient_data = session.exec(statement).first()

    if not patient_data:
        raise HTTPException(status_code=404, detail="Patient data not found")

    return {
        "id": str(patient_data.id),
        "filename": patient_data.filename,
        "filenames": json.loads(patient_data.filename) if patient_data.filename.startswith("[") else [patient_data.filename],
        "extracted_data": json.loads(patient_data.extracted_data),
        "created_at": patient_data.created_at,
    }


@router.delete("/{patientId}")
def delete_patient_data(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    patientId: str,
) -> dict[str, str]:
    """
    Delete patient data by ID.
    """

    try:
        patient_uuid = UUID(patientId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid patient ID format")

    statement = (
        select(PDFExtractedData)
        .where(PDFExtractedData.id == patient_uuid)
        .where(PDFExtractedData.owner_id == current_user.id)
    )

    patient_data = session.exec(statement).first()

    if not patient_data:
        raise HTTPException(status_code=404, detail="Patient data not found")

    session.delete(patient_data)
    session.commit()

    return {"message": "Patient data deleted successfully"}
