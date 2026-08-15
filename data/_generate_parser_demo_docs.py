"""Generate sample_policy.docx for the parsers demo.

Creates a structured policy document with headings, paragraphs, and a
coverage table — enough content to show DocxParser's heading conversion
and table rendering.

Run:
    python3 data/_generate_parser_demo_docs.py
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def _add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
    for row_data in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row_data):
            row_cells[i].text = val


def generate(output_path: Path) -> None:
    doc = Document()

    doc.add_heading("Blue Shield Health Insurance Policy", level=1)

    doc.add_paragraph("Policy Number: BSH-2024-004521")
    doc.add_paragraph("Effective Date: January 1, 2024")
    doc.add_paragraph("Policyholder: Jane Doe")

    doc.add_heading("1. Eligibility", level=2)

    doc.add_heading("1.1 Who Is Eligible", level=3)
    doc.add_paragraph(
        "Coverage under this policy is available to individuals who meet all of the "
        "following criteria: are between the ages of 18 and 65 at the time of enrollment, "
        "are legal residents of the United States, have not been diagnosed with a "
        "pre-existing condition in the 12 months prior to enrollment, and have submitted "
        "a completed application form along with the required premium payment."
    )

    doc.add_heading("1.2 Dependents", level=3)
    doc.add_paragraph(
        "Eligible policyholders may enroll their spouse or domestic partner and dependent "
        "children under the age of 26. Dependent children who are full-time students "
        "remain eligible until the age of 26 regardless of their state of residence. "
        "Documentation of dependent status must be provided within 30 days of enrollment."
    )

    doc.add_heading("2. Coverage Benefits", level=2)
    doc.add_paragraph(
        "This policy provides the following categories of coverage. Benefit limits and "
        "cost-sharing amounts are shown in the table below."
    )

    _add_table(doc,
        headers=["Benefit", "Annual Limit", "Deductible", "Copay"],
        rows=[
            ["Preventive Care",     "Unlimited",   "$0",       "$0"],
            ["Primary Care Visit",  "Unlimited",   "$500",     "$25"],
            ["Specialist Visit",    "Unlimited",   "$500",     "$50"],
            ["Emergency Room",      "Unlimited",   "$500",     "$150"],
            ["Prescription Drugs",  "$10,000",     "$250",     "$15–$45"],
            ["Mental Health",       "60 visits",   "$500",     "$25"],
            ["Hospitalization",     "Unlimited",   "$1,000",   "20% coinsurance"],
        ]
    )

    doc.add_heading("3. Exclusions", level=2)
    doc.add_paragraph(
        "This policy does not cover: cosmetic surgery not required for medical reasons, "
        "experimental treatments not approved by a recognised medical authority, "
        "injuries sustained while committing a felony, or services received outside "
        "the United States except in a medical emergency."
    )

    doc.add_heading("4. Premiums and Payment", level=2)

    doc.add_heading("4.1 Monthly Premium", level=3)
    doc.add_paragraph(
        "The monthly premium for individual coverage is $320.00. Family coverage "
        "is $890.00 per month. Premiums are due on the first of each month. "
        "A grace period of 30 days is provided before coverage lapses."
    )

    doc.add_heading("4.2 Premium Schedule", level=3)
    _add_table(doc,
        headers=["Coverage Tier", "Monthly Premium", "Annual Premium"],
        rows=[
            ["Individual",          "$320.00",   "$3,840.00"],
            ["Individual + Spouse", "$610.00",   "$7,320.00"],
            ["Family",              "$890.00",   "$10,680.00"],
        ]
    )

    doc.add_heading("5. Claims", level=2)
    doc.add_paragraph(
        "Claims must be submitted within 180 days of the date of service. Submit "
        "claims electronically via the member portal or by mail to the address shown "
        "on your insurance card. Include itemised bills and the treating provider's "
        "NPI number with each submission."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"Written: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_policy.docx"
    generate(out)
