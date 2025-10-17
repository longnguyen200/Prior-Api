import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import fitz  # PyMuPDF
from fastapi import APIRouter, HTTPException
from google import genai
from google.genai import types
from sqlmodel import select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    ApprovalReason,
    PDFExtractedData,
    PolicyEvaluation,
    PolicyEvaluationRequest,
    PolicyEvaluationResponse,
    PolicyEvaluationResult,
)

router = APIRouter(prefix="/policy-evaluation", tags=["policy-evaluation"])
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def load_policy_documents(policy_names: list[str] | None = None) -> str:
    """Load medical policy documents from the policy directory."""
    policy_dir = Path(__file__).parent.parent.parent / "policy"
    
    # Default policy files
    default_policies = [
        "Medical_Policy_Guidelines_Heel_Injury.pdf",
        "Medical_Policy_Guidelines_Knee_Injury.pdf",
        "Medical_Policy_Guidelines_Shoulder_Injury.pdf",
    ]
    
    # Use specified policies or all default policies
    policies_to_load = policy_names if policy_names else default_policies
    
    policy_texts = []
    for policy_name in policies_to_load:
        policy_path = policy_dir / policy_name
        if policy_path.exists():
            logger.info(f"Loading policy: {policy_name}")
            text = extract_text_from_pdf(policy_path)
            if text:
                policy_texts.append(f"=== POLICY DOCUMENT: {policy_name} ===\n\n{text}")
        else:
            logger.warning(f"Policy file not found: {policy_path}")
    
    return "\n\n" + "="*80 + "\n\n".join(policy_texts)


def evaluate_with_gemini(
    patient_data: dict[str, Any],
    policy_documents: str,
    api_key: str,
    model_name: str,
) -> PolicyEvaluationResult:
    """
    Evaluate patient data against medical policy using Gemini.
    
    Args:
        patient_data: Extracted patient data
        policy_documents: Combined text of policy documents
        api_key: Gemini API key
        model_name: Model name to use
        
    Returns:
        PolicyEvaluationResult with structured evaluation
    """
    # Extract key information for evaluation
    diagnoses = []
    procedures = []
    
    if "diagnosis" in patient_data:
        diagnoses = [
            f"{d.get('condition', '')} ({d.get('icd_code', 'N/A')})"
            for d in patient_data["diagnosis"]
        ]
    
    if "treatment_plan" in patient_data and "orders" in patient_data["treatment_plan"]:
        procedures = [
            order.get("description", "")
            for order in patient_data["treatment_plan"]["orders"]
            if order.get("order_type") == "Procedure"
        ]
    
    # Create detailed evaluation prompt
    system_prompt = f"""You are a medical policy compliance evaluator for prior authorization.

Your task is to evaluate whether the patient's medical request meets the approval requirements based on the provided medical policy documents.

### POLICY DOCUMENTS
{policy_documents}

### EVALUATION CRITERIA
Evaluate the following aspects:
1. **Conservative Treatment Duration**: Has the patient completed the minimum duration of conservative treatment?
2. **Required Conservative Therapies**: Has the patient tried required therapies (NSAIDs, PT, orthotics, injections, etc.)?
3. **Diagnostic Confirmation**: Are required imaging/diagnostic tests documented?
4. **Pain/Functional Limitations**: Do symptoms meet severity thresholds?
5. **Medical Necessity**: Is the procedure medically necessary based on documentation?
6. **Contraindications**: Are there any contraindications or exclusions?
7. **Investigational Procedures**: Are any requested procedures investigational or non-covered (e.g., PRP)?

### INSTRUCTIONS
1. Review the patient data thoroughly
2. Compare against each policy criterion
3. For each criterion, determine: met, not_met, or insufficient_data
4. Provide specific reasons with evidence from the patient data
5. If information is missing, list what's needed and generate specific questions
6. Provide a final determination: approved, denied, or needs_more_info
7. Include confidence score (0-1) based on completeness of data

### OUTPUT REQUIREMENTS
Return a structured evaluation with:
- approval_reasons: List reasons that support approval
- denial_reasons: List reasons that support denial
- missing_information: Data points that are not available
- questions_for_provider: Specific questions to ask if data is missing
- policy_references: Citations to relevant policy sections
- summary: Overall evaluation summary
- final_determination: approved, denied, or needs_more_info
- confidence_score: 0.0 to 1.0

Be thorough, objective, and cite specific policy requirements.
"""

    patient_summary = f"""
### PATIENT REQUEST SUMMARY
**Procedures Requested**: {', '.join(procedures) if procedures else 'Not specified'}
**Diagnoses**: {', '.join(diagnoses) if diagnoses else 'Not specified'}

### COMPLETE PATIENT DATA
{json.dumps(patient_data, indent=2, ensure_ascii=False)}
"""

    # Initialize Gemini client
    client = genai.Client(api_key=api_key, vertexai=True)
    
    # Define schema for structured output
    response_schema = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "final_determination": {
                    "type": "string",
                    "enum": ["approved", "denied", "needs_more_info"],
                    "description": "Final decision on the prior authorization request"
                },
                "confidence_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence level in the evaluation (0-1)"
                },
                "approval_reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["met", "not_met", "insufficient_data"]
                            },
                            "summary": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["criterion", "status", "summary"]
                    }
                },
                "denial_reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["met", "not_met", "insufficient_data"]
                            },
                            "summary": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["criterion", "status", "summary"]
                    }
                },
                "missing_information": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "policy_references": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "summary": {"type": "string"},
                "questions_for_provider": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "final_determination",
                "confidence_score",
                "approval_reasons",
                "denial_reasons",
                "missing_information",
                "policy_references",
                "summary",
                "questions_for_provider"
            ]
        },
        system_instruction=system_prompt,
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    
    # Generate evaluation
    response = client.models.generate_content(
        model=model_name,
        contents=patient_summary,
        config=response_schema,
    )
    
    # Parse response
    response_text = response.text.strip()
    result_dict = json.loads(response_text)
    
    # Convert to PolicyEvaluationResult
    approval_reasons = [
        ApprovalReason(**reason) for reason in result_dict.get("approval_reasons", [])
    ]
    denial_reasons = [
        ApprovalReason(**reason) for reason in result_dict.get("denial_reasons", [])
    ]
    
    return PolicyEvaluationResult(
        final_determination=result_dict["final_determination"],
        confidence_score=result_dict["confidence_score"],
        approval_reasons=approval_reasons,
        denial_reasons=denial_reasons,
        missing_information=result_dict.get("missing_information", []),
        policy_references=result_dict.get("policy_references", []),
        summary=result_dict["summary"],
        questions_for_provider=result_dict.get("questions_for_provider", []),
    )


@router.post("/evaluate", response_model=PolicyEvaluationResponse)
async def evaluate_policy(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    request: PolicyEvaluationRequest,
    save_to_db: bool = True,
) -> Any:
    """
    Evaluate extracted patient data against medical policy for prior authorization.
    
    - **extracted_data_id**: ID of the extracted patient data record
    - **policy_documents**: Optional list of specific policy documents to use
    - **save_to_db**: If True, save evaluation result to database
    
    Returns structured evaluation with approval/denial reasons and questions if needed.
    """
    # Validate API key
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured in environment",
        )
    
    try:
        # Get extracted patient data
        try:
            patient_uuid = UUID(request.extracted_data_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid extracted_data_id format"
            )
        
        statement = select(PDFExtractedData).where(
            PDFExtractedData.id == patient_uuid,
            PDFExtractedData.owner_id == current_user.id,
        )
        
        patient_record = session.exec(statement).first()
        
        if not patient_record:
            raise HTTPException(
                status_code=404,
                detail="Patient data not found or access denied",
            )
        
        # Parse patient data
        patient_data = json.loads(patient_record.extracted_data)
        
        # Load policy documents
        logger.info("Loading policy documents...")
        policy_documents = load_policy_documents(request.policy_documents)
        
        if not policy_documents:
            raise HTTPException(
                status_code=500,
                detail="Failed to load policy documents",
            )
        
        # Evaluate with Gemini
        logger.info("Evaluating policy compliance with Gemini...")
        evaluation_result = evaluate_with_gemini(
            patient_data=patient_data,
            policy_documents=policy_documents,
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.GEMINI_MODEL_NAME,
        )
        
        # Save to database if requested
        saved_id = None
        if save_to_db:
            policy_eval = PolicyEvaluation(
                patient_data=patient_record.extracted_data,
                pa_form_data=json.dumps(patient_data, ensure_ascii=False),  # Can be separated if needed
                evaluation_result=evaluation_result.model_dump_json(),
                final_determination=evaluation_result.final_determination,
                confidence_score=evaluation_result.confidence_score,
                owner_id=current_user.id,
            )
            session.add(policy_eval)
            session.commit()
            session.refresh(policy_eval)
            saved_id = str(policy_eval.id)
            logger.info(f"Saved policy evaluation with ID: {policy_eval.id}")
        
        return PolicyEvaluationResponse(
            success=True,
            evaluation_result=evaluation_result,
            evaluation_id=saved_id,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating policy: {e}")
        return PolicyEvaluationResponse(
            success=False,
            error=str(e),
        )


@router.get("/{evaluation_id}")
def get_evaluation(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    evaluation_id: str,
) -> Any:
    """Get policy evaluation by ID."""
    try:
        eval_uuid = UUID(evaluation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid evaluation ID format")
    
    statement = select(PolicyEvaluation).where(
        PolicyEvaluation.id == eval_uuid,
        PolicyEvaluation.owner_id == current_user.id,
    )
    
    evaluation = session.exec(statement).first()
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return {
        "id": str(evaluation.id),
        "evaluation_result": json.loads(evaluation.evaluation_result),
        "final_determination": evaluation.final_determination,
        "confidence_score": evaluation.confidence_score,
        "created_at": evaluation.created_at,
    }


@router.get("/")
def list_evaluations(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List all policy evaluations for current user."""
    statement = (
        select(PolicyEvaluation)
        .where(PolicyEvaluation.owner_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(PolicyEvaluation.created_at.desc())  # type: ignore
    )
    
    results = session.exec(statement).all()
    
    return {
        "data": [
            {
                "id": str(item.id),
                "final_determination": item.final_determination,
                "confidence_score": item.confidence_score,
                "evaluation_result": json.loads(item.evaluation_result),
                "created_at": item.created_at,
            }
            for item in results
        ],
        "count": len(results),
        "skip": skip,
        "limit": limit,
    }

