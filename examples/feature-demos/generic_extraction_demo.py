"""
Generic field extraction without a classifier
==============================================
Shows how to run LLMFieldExtractor on a corpus you haven't typed yet —
no KeywordClassifier required. The pipeline assigns document_type="unknown"
and confidence=0.0 automatically; extracted fields are still stored and
queryable via DOCUMENT.FIND.

Story:
  A research team dumps a mix of internal documents into the pipeline:
  lab notebooks, equipment logs, and incident reports. They don't have
  a classifier trained on these types yet. They still want LLM-extracted
  fields searchable immediately.

What this demo covers:
  * field_extractor without classifier — document_type defaults to "unknown"
  * Fields extracted and stored in the record just like the typed path
  * DOCUMENT.FIND filters work identically on "unknown" type records
  * Adding a classifier later gives you typed records + extracted fields
  * Side-by-side: no classifier vs with classifier

Run:
    python3 examples/feature-demos/generic_extraction_demo.py

    # With real LLM extraction:
    OPENAI_API_KEY=sk-... python3 examples/feature-demos/generic_extraction_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blue_data_doc_source import DocumentPipeline, LLMFieldExtractor, KeywordClassifier

DIV  = "=" * 72
DIV2 = "-" * 72


# ---------------------------------------------------------------------------
# Sample texts — mixed, unknown document types (no headers the classifier
# would recognize as contract/policy/resume/report/manual)
# ---------------------------------------------------------------------------

LAB_NOTEBOOK_TEXT = """
Experiment: Membrane Diffusion Study
Date: 2026-07-22
Researcher: Dr. Anika Patel
Lab Location: Building C, Room 204
Protocol: Standard diffusion assay, 6-hour incubation at 37°C
Sample ID: MEM-2026-047
Observations: Diffusion rate 12% higher than baseline. pH stable at 7.4.
    Membrane integrity confirmed post-assay.
Result: PASS
Next Steps: Repeat with membrane lot MEM-2026-048 for comparison.
"""

EQUIPMENT_LOG_TEXT = """
Equipment Maintenance Log

Equipment: Centrifuge Model CF-800
Asset ID: EQ-20190342
Service Date: 2026-07-30
Technician: James Ortega
Hours Since Last Service: 342
Issues Found: Drive belt showing wear; replaced.
    Rotor balance re-calibrated to within 0.01g tolerance.
Parts Replaced: Drive belt (part #CF800-BELT-22)
Next Service Due: 2026-10-30 or 400 hours, whichever comes first
Certification Status: ACTIVE
"""

INCIDENT_REPORT_TEXT = """
INCIDENT REPORT

Report ID: INC-2026-0088
Date of Incident: 2026-08-01
Location: Cold Storage Room, Building A
Reported By: Technician Mira Santos
Incident Type: Equipment Failure
Severity: Medium

Description: Walk-in freezer unit failed at 02:14. Temperature rose from
-20°C to -8°C over 4 hours before alarm triggered. Approximately 15 sample
boxes were exposed to elevated temperature. Facilities notified at 06:30.
Freezer repaired by 09:00. Sample integrity under review.

Immediate Actions Taken:
  - Samples transferred to backup freezer (Building B, Room 102)
  - Facilities maintenance dispatched
  - Lab manager and PI notified

Root Cause: Compressor failure due to refrigerant leak.
Status: Resolved
Follow-Up Required: Sample viability assessment by 2026-08-05
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
    _banner("Generic field extraction — field_extractor without a classifier")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    print()
    print("  Three document types the pipeline has never seen:")
    print("    lab_notebook, equipment_log, incident_report")
    print()
    print("  No classifier. No extra_types. LLM discovers fields as-is.")
    print()
    if has_openai:
        print("  OPENAI_API_KEY detected — LLM extraction enabled.")
    else:
        print("  No OPENAI_API_KEY — showing pipeline setup and filter syntax.")
        print("  Set OPENAI_API_KEY to see actual extracted fields.")

    # ── Step 1: Pipeline with field_extractor only ────────────────────────
    _step(1, "Pipeline setup — field_extractor alone, no classifier")
    print()
    print("  pipeline = DocumentPipeline(")
    print("      field_extractor=LLMFieldExtractor(),")
    print("      # No classifier= argument")
    print("  )")
    print()
    print("  For each document ingested:")
    print("    document_type = 'unknown'   (no classifier to determine type)")
    print("    confidence    = 0.0")
    print("    fields        = { ... }      (LLM extracts whatever it finds)")
    print()

    if has_openai:
        pipeline = DocumentPipeline(
            chunk_size=250, overlap=25,
            field_extractor=LLMFieldExtractor(),
        )
    else:
        pipeline = None

    # ── Step 2: Ingest three unknown-type documents ───────────────────────
    _step(2, "Ingest three mixed documents")
    print()

    docs = [
        ("lab_notebook_2026.txt",  "Lab Notebook",      LAB_NOTEBOOK_TEXT),
        ("equipment_log_cf800.txt", "Equipment Log",     EQUIPMENT_LOG_TEXT),
        ("incident_INC-2026-0088.txt", "Incident Report", INCIDENT_REPORT_TEXT),
    ]

    for fname, label, text in docs:
        print(f"  pipeline.ingest_text(<{label}>, file_name={fname!r})")
        if has_openai:
            pipeline.ingest_text(text, file_name=fname)

    if has_openai:
        records = pipeline.find_records()
        print()
        print(f"  Documents stored: {len(records)}")
        print()
        for r in records:
            print(f"  ┌── {r.get('source_name', r['document_id'][:12])}")
            print(f"  │   document_type = {r['document_type']!r}")
            print(f"  │   confidence    = {r['confidence']}")
            print(f"  │   fields ({len(r['fields'])} extracted):")
            for k, v in list(r["fields"].items())[:5]:
                v_str = str(v)
                if len(v_str) > 55:
                    v_str = v_str[:52] + "..."
                print(f"  │     {k:<22} {v_str!r}")
            if len(r["fields"]) > 5:
                print(f"  │     ... +{len(r['fields']) - 5} more")
            print(f"  └──")
            print()
    else:
        print()
        print("  (Set OPENAI_API_KEY to see extracted fields)")

    # ── Step 3: DOCUMENT.FIND still works ────────────────────────────────
    _step(3, "DOCUMENT.FIND — query across unknown-type records")
    print()
    print("  find_records() and filter syntax work identically for unknown-type records.")
    print("  Fields are LLM-discovered, so key names depend on what the LLM extracted.")
    print()

    if has_openai:
        print("  All records:")
        all_r = pipeline.find_records()
        print(f"    pipeline.find_records()  ->  {len(all_r)} records")
        print()

        # Try filtering on a field that should exist in the incident report
        print("  Filtering for Severity field (incident report):")
        severity_r = pipeline.find_records({"fields.Severity": {"exists": True}})
        if severity_r:
            print(f"    filter: {{'fields.Severity': {{'exists': True}}}}")
            for r in severity_r:
                print(f"    -> {r.get('source_name', r['document_id'][:16])} "
                      f"  Severity={r['fields'].get('Severity')!r}")
        else:
            # Field name may differ by LLM run — show what fields are available
            print("    (LLM may have used a different field name — "
                  "showing all field keys from each record:)")
            for r in all_r:
                print(f"    {r.get('source_name', r['document_id'][:16])}: "
                      + ", ".join(r["fields"].keys()))
        print()

        print("  All available field keys per document:")
        for r in all_r:
            doc_name = r.get("source_name", r["document_id"][:16])
            keys = list(r["fields"].keys())
            print(f"    {doc_name}: {keys}")
    else:
        print("  Showing filter syntax — set OPENAI_API_KEY to see real results:")
        print()
        print("    # All records (document_type='unknown')")
        print("    pipeline.find_records()")
        print()
        print("    # Records where the LLM found a 'Severity' field")
        print("    pipeline.find_records({'fields.Severity': {'exists': True}})")
        print()
        print("    # Records where Result is 'PASS'")
        print("    pipeline.find_records({'fields.Result': 'PASS'})")
        print()
        print("    # Records where status is resolved")
        print("    pipeline.find_records({'fields.Status': 'Resolved'})")

    # ── Step 4: Adding a classifier later ────────────────────────────────
    _step(4, "Adding a classifier later — typed records + extracted fields")
    print()
    print("  Once you've defined extra_types, add a classifier alongside the extractor:")
    print()
    print("    EXTRA_TYPES = {")
    print("        'lab_notebook': [")
    print("            'experiment', 'researcher', 'protocol', 'sample id',")
    print("            'observations', 'assay',")
    print("        ],")
    print("        'equipment_log': [")
    print("            'maintenance', 'equipment', 'asset id', 'technician',")
    print("            'service', 'parts replaced', 'certification',")
    print("        ],")
    print("        'incident_report': [")
    print("            'incident', 'severity', 'root cause', 'immediate actions',")
    print("            'follow-up', 'compressor',")
    print("        ],")
    print("    }")
    print()
    print("    pipeline = DocumentPipeline(")
    print("        classifier=KeywordClassifier(extra_types=EXTRA_TYPES),")
    print("        field_extractor=LLMFieldExtractor(extra_types={")
    print("            'lab_notebook'    : 'Extract: researcher, date, sample_id, result, next_steps',")
    print("            'equipment_log'   : 'Extract: asset_id, technician, service_date, next_service_due, status',")
    print("            'incident_report' : 'Extract: report_id, severity, incident_date, root_cause, status',")
    print("        }),")
    print("    )")
    print()
    print("  Now each record gets:")
    print("    document_type = 'lab_notebook' | 'equipment_log' | 'incident_report'")
    print("    confidence    = 0.0–1.0  (keyword score)")
    print("    fields        = { ... }   (type-specific extraction)")
    print()

    if has_openai:
        EXTRA_TYPES = {
            "lab_notebook": [
                "experiment", "researcher", "protocol", "sample id",
                "observations", "assay",
            ],
            "equipment_log": [
                "maintenance", "equipment", "asset id", "technician",
                "service", "parts replaced", "certification",
            ],
            "incident_report": [
                "incident", "severity", "root cause", "immediate actions",
                "follow-up", "compressor",
            ],
        }

        pipeline_typed = DocumentPipeline(
            chunk_size=250, overlap=25,
            classifier=KeywordClassifier(extra_types=EXTRA_TYPES),
            field_extractor=LLMFieldExtractor(extra_types={
                "lab_notebook":    "Extract: researcher, date, sample_id, result, next_steps",
                "equipment_log":   "Extract: asset_id, technician, service_date, next_service_due, status",
                "incident_report": "Extract: report_id, severity, incident_date, root_cause, status",
            }),
        )
        for fname, label, text in docs:
            pipeline_typed.ingest_text(text, file_name=fname)

        typed_records = pipeline_typed.find_records()
        print("  With classifier:")
        print()
        for r in typed_records:
            print(f"  ┌── {r.get('source_name', r['document_id'][:12])}")
            print(f"  │   document_type = {r['document_type']!r}  "
                  f"confidence={r['confidence']:.2f}")
            for k, v in list(r["fields"].items())[:4]:
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:47] + "..."
                print(f"  │   {k:<22} {v_str!r}")
            print(f"  └──")
            print()

    # ── Step 5: When to use which path ───────────────────────────────────
    _step(5, "Decision guide — pick your path")
    print()
    print("  Do you need extracted fields?")
    print("    │")
    print("    ├─ No  → KeywordClassifier() alone")
    print("    │         document_type on every chunk, filterable in KNN — zero API cost")
    print("    │")
    print("    └─ Yes → Do you already have a schema (Pydantic / TypedDict / JSON Schema)?")
    print("               │")
    print("               ├─ No  → LLMFieldExtractor() alone          ← start here")
    print("               │         LLM names fields freely — inspect output once:")
    print("               │           find_records() → learn field names → write filters")
    print("               │         When names are stable → move to StructuredFieldExtractor")
    print("               │")
    print("               └─ Yes → Do you also need document_type on each chunk?")
    print("                          │")
    print("                          ├─ No  → StructuredFieldExtractor(schema)")
    print("                          │         schema-defined names, Pydantic-validated")
    print("                          │")
    print("                          └─ Yes → KeywordClassifier() + StructuredFieldExtractor(schema)")
    print("                                    detected document_type + schema-validated fields")

    # ── Summary ───────────────────────────────────────────────────────────
    _banner("Summary")
    print()
    print("  field_extractor works without a classifier:")
    print()
    print("    pipeline = DocumentPipeline(")
    print("        field_extractor=LLMFieldExtractor(),")
    print("    )")
    print()
    print("  The pipeline assigns document_type='unknown', confidence=0.0")
    print("  and runs the field extractor on every ingested document.")
    print("  Extracted fields are stored in the record and fully queryable.")
    print()
    print("  This is the fastest path to searchable structured fields from")
    print("  an unfamiliar or heterogeneous corpus — zero classifier setup.")
    print()
    if not has_openai:
        print("  To see real extraction output:")
        print("    OPENAI_API_KEY=sk-... python3 examples/feature-demos/generic_extraction_demo.py")
        print()


if __name__ == "__main__":
    run()
