"""
StructuredFieldExtractor — schema-driven field extraction
=========================================================
Shows how to extract document fields against a caller-supplied schema
(Pydantic model, TypedDict, or JSON Schema dict) without writing a
manual field spec string.

Story:
  A legal ops team processes employment contracts and lease agreements —
  neither of which is one of the five built-in document types. They want
  typed, validated field extraction from day one without hardcoding anything.

What this demo covers:
  * StructuredFieldExtractor with a Pydantic model (+ output validation)
  * StructuredFieldExtractor with a TypedDict
  * StructuredFieldExtractor with a JSON Schema dict
  * Prompt auto-generated from the schema — no field spec string needed
  * Pydantic model_validate coerces and validates LLM output
  * DOCUMENT.FIND filters work identically for schema-extracted fields

Run:
    python3 examples/feature-demos/structured_extractor_demo.py

    # With real LLM extraction:
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/structured_extractor_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data_doc_source import DocumentPipeline, StructuredFieldExtractor

DIV  = "=" * 72
DIV2 = "-" * 72

# ---------------------------------------------------------------------------
# Sample document texts (inline — no PDFs needed)
# ---------------------------------------------------------------------------

EMPLOYMENT_TEXT = """
EMPLOYMENT AGREEMENT

This Employment Agreement is entered into between NovaTech Solutions Inc.
(the "Employer") and Sarah Chen (the "Employee").

Position: Senior Software Engineer
Start Date: September 1, 2026
Base Salary: $145,000 per year, paid bi-weekly
Notice Period: 60 days written notice required from either party
Non-Compete: Employee agrees not to work for direct competitors for
12 months following termination within the same metropolitan area.
Jurisdiction: State of California
"""

LEASE_TEXT = """
RESIDENTIAL LEASE AGREEMENT

Landlord: Greenview Properties LLC
Tenant: Marcus Webb
Property Address: 742 Evergreen Terrace, Portland, OR 97201
Monthly Rent: $2,400.00 due on the 1st of each month
Lease Term: 12 months commencing October 1, 2026
Security Deposit: $4,800.00 (two months rent)
Late Fee: $75 after 5-day grace period
Utilities: Tenant responsible for electricity and internet.
"""


def _banner(title: str) -> None:
    print()
    print(DIV)
    print(title)
    print(DIV)


def _step(n: int, title: str) -> None:
    print()
    print(DIV2)
    print(f"Step {n} -- {title}")
    print(DIV2)


def run() -> None:
    _banner("StructuredFieldExtractor — schema-driven field extraction")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Extracts fields from any document type using a caller-supplied schema.")
    print("  No manual field spec string. No hardcoded document types.")
    print()
    if has_openai:
        print("  OPENAI_API_KEY detected — LLM extraction enabled.")
    else:
        print("  No OPENAI_API_KEY — showing auto-generated prompts only.")
        print("  Set OPENAI_API_KEY to see actual extracted fields.")

    # ── Step 1: Pydantic model ────────────────────────────────────────────
    _step(1, "Pydantic model — auto-generated prompt + validated output")
    print()
    print("  Define your schema as a Pydantic model:")
    print()
    print("    from pydantic import BaseModel")
    print()
    print("    class EmploymentContract(BaseModel):")
    print("        employee_name: str")
    print("        employer_name: str")
    print("        start_date: str")
    print("        base_salary: str")
    print("        notice_period: str")
    print("        non_compete: bool")
    print("        jurisdiction: str")
    print()

    try:
        from pydantic import BaseModel

        class EmploymentContract(BaseModel):
            employee_name: str
            employer_name: str
            start_date: str
            base_salary: str
            notice_period: str
            non_compete: bool
            jurisdiction: str

        extractor = StructuredFieldExtractor(schema=EmploymentContract)
        print(f"  Auto-generated field spec:")
        print(f"    {extractor._field_spec}")
        print()

        if has_openai:
            pipeline = DocumentPipeline(
                chunk_size=200, overlap=20,
                field_extractor=StructuredFieldExtractor(schema=EmploymentContract),
            )
            chunks = pipeline.ingest_text(EMPLOYMENT_TEXT, file_name="employment_chen.txt")
            record = pipeline.find_records()[0]

            print(f"  Ingested: {len(chunks)} chunks  document_type={record['document_type']!r}")
            print(f"  Pydantic-validated fields:")
            for k, v in record["fields"].items():
                print(f"    {k:<20} {v!r}")
        else:
            print("  (Set OPENAI_API_KEY to see extracted and validated fields)")

    except ImportError:
        print("  pydantic not installed — skipping Pydantic example.")
        print("  Install with: pip install pydantic")

    # ── Step 2: TypedDict ─────────────────────────────────────────────────
    _step(2, "TypedDict — prompt generated from annotations")
    print()
    print("  Use a TypedDict when you don't need Pydantic validation:")
    print()
    print("    class LeaseInfo(TypedDict):")
    print("        landlord: str")
    print("        tenant: str")
    print("        property_address: str")
    print("        monthly_rent: str")
    print("        lease_term_months: int")
    print("        security_deposit: str")
    print()

    class LeaseInfo(TypedDict):
        landlord: str
        tenant: str
        property_address: str
        monthly_rent: str
        lease_term_months: int
        security_deposit: str

    extractor_td = StructuredFieldExtractor(schema=LeaseInfo)
    print(f"  Auto-generated field spec:")
    print(f"    {extractor_td._field_spec}")
    print()

    if has_openai:
        pipeline_td = DocumentPipeline(
            chunk_size=200, overlap=20,
            field_extractor=StructuredFieldExtractor(schema=LeaseInfo),
        )
        pipeline_td.ingest_text(LEASE_TEXT, file_name="lease_webb.txt")
        record_td = pipeline_td.find_records()[0]

        print(f"  Extracted fields (no Pydantic validation — returned as-is):")
        for k, v in record_td["fields"].items():
            print(f"    {k:<22} {v!r}")
    else:
        print("  (Set OPENAI_API_KEY to see extracted fields)")

    # ── Step 3: JSON Schema dict ──────────────────────────────────────────
    _step(3, "JSON Schema dict — for schema-first or API-driven workflows")
    print()
    print("  Pass a JSON Schema dict directly — useful when schema comes from")
    print("  an external registry or is constructed programmatically:")
    print()

    VENDOR_AGREEMENT_SCHEMA = {
        "type": "object",
        "properties": {
            "vendor_name":    {"type": "string"},
            "client_name":    {"type": "string"},
            "contract_value": {"type": "string"},
            "payment_terms":  {"type": "string"},
            "sla_uptime":     {"type": "string"},
            "auto_renewal":   {"type": "boolean"},
        },
    }

    print("    schema = {")
    print("        'type': 'object',")
    print("        'properties': {")
    for field, info in VENDOR_AGREEMENT_SCHEMA["properties"].items():
        print(f"            {field!r}: {{'type': {info['type']!r}}},")
    print("        },")
    print("    }")
    print()

    extractor_js = StructuredFieldExtractor(schema=VENDOR_AGREEMENT_SCHEMA)
    print(f"  Auto-generated field spec:")
    print(f"    {extractor_js._field_spec}")

    # ── Step 4: DOCUMENT.FIND with schema-extracted fields ────────────────
    _step(4, "DOCUMENT.FIND — same filter API regardless of schema format")
    print()

    if has_openai:
        print("  filter_records() works identically for Pydantic, TypedDict,")
        print("  and JSON Schema extraction results:")
        print()

        # Re-ingest both into one pipeline
        try:
            from pydantic import BaseModel

            class EmploymentContractV2(BaseModel):
                employee_name: str
                employer_name: str
                start_date: str
                base_salary: str
                non_compete: bool

            pipeline_combined = DocumentPipeline(
                chunk_size=200, overlap=20,
                field_extractor=StructuredFieldExtractor(schema=EmploymentContractV2),
            )
            pipeline_combined.ingest_text(EMPLOYMENT_TEXT, file_name="employment_chen.txt")
            pipeline_combined.ingest_text(LEASE_TEXT, file_name="lease_webb.txt")

            all_records = pipeline_combined.find_records()
            print(f"  find_records()  ->  {len(all_records)} record(s)  "
                  f"(all document_type='unknown', fields from schema)")
            print()

            for r in all_records:
                print(f"  [{r['document_id']}]  file inferred from fields:")
                for k, v in list(r["fields"].items())[:3]:
                    print(f"    {k:<20} {v!r}")
                if len(r["fields"]) > 3:
                    print(f"    ... +{len(r['fields']) - 3} more fields")
                print()
        except ImportError:
            pass
    else:
        print("  Showing filter syntax — set OPENAI_API_KEY to see real results:")
        print()
        print("    pipeline.find_records({'fields.non_compete': True})")
        print("    pipeline.find_records({'fields.monthly_rent': {'contains': '$2'}})")
        print("    pipeline.find_records({'fields.lease_term_months': {'gte': 12}})")

    # ── Step 5: Comparison — manual spec vs schema ────────────────────────
    _step(5, "Comparison: LLMFieldExtractor(extra_types=...) vs StructuredFieldExtractor")
    print()
    print("  LLMFieldExtractor(extra_types=...)  -- manual string spec:")
    print()
    print("    extractor = LLMFieldExtractor(extra_types={")
    print("        'employment': (")
    print("            'Extract: employee_name (string), employer_name (string),")
    print("             start_date (ISO date), base_salary (string), non_compete (bool)'")
    print("        )")
    print("    })")
    print("    # Requires pairing with KeywordClassifier(extra_types={'employment': [...]})")
    print("    # Returns raw dict — no type validation")
    print()
    print("  StructuredFieldExtractor(schema=EmploymentContract)  -- schema-driven:")
    print()
    print("    extractor = StructuredFieldExtractor(schema=EmploymentContract)")
    print("    # Prompt auto-generated from model fields and types")
    print("    # Pydantic model_validate() coerces and validates output")
    print("    # Works without a classifier — document_type='unknown'")
    print("    # Add classifier= to get typed document_type alongside schema fields")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  StructuredFieldExtractor extracts fields from any document type")
    print("  using a caller-supplied schema — no manual spec string required:")
    print()
    print("    Pydantic model  ->  auto-prompt + model_validate() output coercion")
    print("    TypedDict       ->  auto-prompt from __annotations__")
    print("    JSON Schema     ->  auto-prompt from properties dict")
    print()
    print("  No classifier required:")
    print("    pipeline = DocumentPipeline(")
    print("        field_extractor=StructuredFieldExtractor(schema=MyModel),")
    print("    )")
    print("    # document_type='unknown', confidence=0.0")
    print("    # fields validated against MyModel")
    print()
    print("  Pair with a classifier to get typed document_type + schema fields:")
    print("    pipeline = DocumentPipeline(")
    print("        classifier=KeywordClassifier(),")
    print("        field_extractor=StructuredFieldExtractor(schema=MyModel),")
    print("    )")
    print()
    if not has_openai:
        print("  To see real extraction output:")
        print("    OPENAI_API_KEY=sk-... python3 examples/feature-demos/structured_extractor_demo.py")
        print()


if __name__ == "__main__":
    run()
