#!/usr/bin/env python3
"""
Example script demonstrating the Prior Authorization Policy Evaluation System.

This script shows the complete workflow:
1. Extract data from multiple PDF files (Medical Records + PA Form)
2. Evaluate against medical policy documents
3. Display structured results with approval/denial reasons
4. Show questions for provider if information is missing

Usage:
    python example_policy_evaluation.py --token YOUR_AUTH_TOKEN
"""

import argparse
import json
from pathlib import Path

import requests


class PolicyEvaluationClient:
    """Client for interacting with the Prior Authorization API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def extract_from_files(
        self, file_paths: list[str], save_to_db: bool = True
    ) -> dict:
        """
        Extract and merge data from multiple PDF files.

        Args:
            file_paths: List of PDF file paths to process
            save_to_db: Whether to save to database

        Returns:
            Extraction response with merged data
        """
        print(f"\n📄 Extracting data from {len(file_paths)} files...")
        for fp in file_paths:
            print(f"  - {Path(fp).name}")

        files = [("files", open(fp, "rb")) for fp in file_paths]

        try:
            response = requests.post(
                f"{self.base_url}/pdf-extraction/extract-batch",
                headers=self.headers,
                files=files,
                params={"save_to_db": save_to_db},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                print(f"✅ Successfully extracted data from {result['total_files']} files")
                print(f"📝 Record ID: {result['id']}")
                return result
            else:
                print(f"❌ Extraction failed: {result.get('error')}")
                return result

        finally:
            # Close file handles
            for _, file_obj in files:
                file_obj.close()

    def evaluate_policy(
        self,
        extracted_data_id: str,
        policy_documents: list[str] | None = None,
        save_to_db: bool = True,
    ) -> dict:
        """
        Evaluate extracted data against medical policy.

        Args:
            extracted_data_id: ID of extracted patient data
            policy_documents: Optional list of specific policy documents
            save_to_db: Whether to save evaluation result

        Returns:
            Evaluation response with structured decision
        """
        print(f"\n🔍 Evaluating against medical policy...")

        request_data = {"extracted_data_id": extracted_data_id}
        if policy_documents:
            request_data["policy_documents"] = policy_documents
            print(f"📋 Using policies: {', '.join(policy_documents)}")

        response = requests.post(
            f"{self.base_url}/policy-evaluation/evaluate",
            headers=self.headers,
            json=request_data,
            params={"save_to_db": save_to_db},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def get_evaluation(self, evaluation_id: str) -> dict:
        """Get evaluation by ID."""
        response = requests.get(
            f"{self.base_url}/policy-evaluation/{evaluation_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def list_evaluations(self, skip: int = 0, limit: int = 10) -> dict:
        """List all evaluations."""
        response = requests.get(
            f"{self.base_url}/policy-evaluation/",
            headers=self.headers,
            params={"skip": skip, "limit": limit},
        )
        response.raise_for_status()
        return response.json()


def print_evaluation_result(evaluation: dict):
    """Pretty print evaluation results."""
    if not evaluation.get("success"):
        print(f"\n❌ Evaluation failed: {evaluation.get('error')}")
        return

    result = evaluation["evaluation_result"]

    # Header
    print("\n" + "=" * 80)
    print("PRIOR AUTHORIZATION EVALUATION RESULT")
    print("=" * 80)

    # Final Determination
    determination = result["final_determination"]
    confidence = result["confidence_score"]

    emoji = {
        "approved": "✅",
        "denied": "❌",
        "needs_more_info": "❓",
    }.get(determination, "⚠️")

    print(f"\n{emoji} FINAL DETERMINATION: {determination.upper()}")
    print(f"📊 Confidence Score: {confidence:.1%}")

    # Summary
    print(f"\n📝 SUMMARY:")
    print(f"{result['summary']}")

    # Approval Reasons
    if result.get("approval_reasons"):
        print(f"\n✅ APPROVAL REASONS ({len(result['approval_reasons'])}):")
        for i, reason in enumerate(result["approval_reasons"], 1):
            status_emoji = {
                "met": "✓",
                "not_met": "✗",
                "insufficient_data": "?",
            }.get(reason["status"], "-")
            print(f"\n  {i}. [{status_emoji}] {reason['criterion']}")
            print(f"     Status: {reason['status']}")
            print(f"     Summary: {reason['summary']}")
            if reason.get("details"):
                print(f"     Details: {reason['details']}")

    # Denial Reasons
    if result.get("denial_reasons"):
        print(f"\n❌ DENIAL REASONS ({len(result['denial_reasons'])}):")
        for i, reason in enumerate(result["denial_reasons"], 1):
            status_emoji = {
                "met": "✓",
                "not_met": "✗",
                "insufficient_data": "?",
            }.get(reason["status"], "-")
            print(f"\n  {i}. [{status_emoji}] {reason['criterion']}")
            print(f"     Status: {reason['status']}")
            print(f"     Summary: {reason['summary']}")
            if reason.get("details"):
                print(f"     Details: {reason['details']}")

    # Missing Information
    if result.get("missing_information"):
        print(f"\n⚠️  MISSING INFORMATION ({len(result['missing_information'])}):")
        for i, info in enumerate(result["missing_information"], 1):
            print(f"  {i}. {info}")

    # Questions for Provider
    if result.get("questions_for_provider"):
        print(f"\n❓ QUESTIONS FOR PROVIDER ({len(result['questions_for_provider'])}):")
        for i, question in enumerate(result["questions_for_provider"], 1):
            print(f"  {i}. {question}")

    # Policy References
    if result.get("policy_references"):
        print(f"\n📚 POLICY REFERENCES:")
        for i, ref in enumerate(result["policy_references"], 1):
            print(f"  {i}. {ref}")

    print("\n" + "=" * 80)

    # Save evaluation ID if available
    if evaluation.get("evaluation_id"):
        print(f"\n💾 Evaluation saved with ID: {evaluation['evaluation_id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Prior Authorization Policy Evaluation Example"
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Authentication token for API access",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost/api/v1",
        help="Base URL for API (default: http://localhost/api/v1)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="PDF files to extract (e.g., Medical_records.pdf PA_Form.pdf)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        help="Specific policy documents to use for evaluation",
    )
    parser.add_argument(
        "--extracted-id",
        help="Use existing extracted data ID (skip extraction step)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent evaluations",
    )

    args = parser.parse_args()

    client = PolicyEvaluationClient(args.base_url, args.token)

    # List evaluations if requested
    if args.list:
        print("\n📋 Listing recent evaluations...")
        result = client.list_evaluations(limit=10)
        print(f"\nFound {result['count']} evaluations:")
        for item in result["data"]:
            print(f"\n  ID: {item['id']}")
            print(f"  Determination: {item['final_determination']}")
            print(f"  Confidence: {item['confidence_score']:.1%}")
            print(f"  Created: {item['created_at']}")
        return

    # Extract data from files or use existing ID
    if args.extracted_id:
        extracted_data_id = args.extracted_id
        print(f"\n✓ Using existing extracted data ID: {extracted_data_id}")
    elif args.files:
        # Verify files exist
        for file_path in args.files:
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")
                return

        extraction_result = client.extract_from_files(args.files)

        if not extraction_result.get("success"):
            print("❌ Extraction failed. Cannot proceed with evaluation.")
            return

        extracted_data_id = extraction_result["id"]

        # Show extracted data summary
        data = extraction_result.get("extracted_data", {})
        if data.get("patient_info"):
            patient = data["patient_info"]
            print(f"\n👤 Patient: {patient.get('full_name', 'N/A')}")
            print(f"   MRN: {patient.get('mrn', 'N/A')}")
            print(f"   DOB: {patient.get('date_of_birth', 'N/A')}")

        if data.get("diagnosis"):
            print(f"\n🩺 Diagnoses:")
            for dx in data["diagnosis"][:3]:  # Show first 3
                print(f"   - {dx.get('condition', 'N/A')} ({dx.get('icd_code', 'N/A')})")

        if data.get("treatment_plan", {}).get("orders"):
            print(f"\n💊 Procedures Requested:")
            for order in data["treatment_plan"]["orders"]:
                if order.get("order_type") == "Procedure":
                    print(f"   - {order.get('description', 'N/A')}")
    else:
        print("❌ Error: Must provide either --files or --extracted-id")
        parser.print_help()
        return

    # Evaluate against policy
    evaluation = client.evaluate_policy(
        extracted_data_id=extracted_data_id,
        policy_documents=args.policies,
    )

    # Print results
    print_evaluation_result(evaluation)


if __name__ == "__main__":
    main()

