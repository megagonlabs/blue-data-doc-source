"""Generate data/sample_policy.pdf from the policy text content.

Run once: python data/_generate_sample_pdf.py
Requires fpdf2: pip install fpdf2
"""
from fpdf import FPDF
from pathlib import Path

SECTIONS = [
    {
        "number": "1",
        "title": "ELIGIBILITY",
        "subsections": [
            ("1.1 Who Is Eligible",
             "Coverage under this policy is available to individuals who meet all of the "
             "following criteria: are between the ages of 18 and 65 at the time of enrollment, "
             "are legal residents of the United States, have not been diagnosed with a "
             "pre-existing condition in the 12 months prior to enrollment, and have submitted "
             "a completed application form along with the required premium payment."),
            ("1.2 Dependents",
             "Eligible policyholders may enroll their spouse or domestic partner and dependent "
             "children under the age of 26. Dependent children who are full-time students "
             "remain eligible until the age of 26 regardless of their state of residence. "
             "Documentation of dependent status must be provided within 30 days of enrollment."),
            ("1.3 Enrollment Periods",
             "Open enrollment runs from November 1 to December 15 each year, with coverage "
             "beginning on January 1 of the following year. Special enrollment periods of 60 "
             "days are available following qualifying life events including marriage, birth or "
             "adoption of a child, loss of other coverage, or relocation to a new service area."),
        ],
    },
    {
        "number": "2",
        "title": "COVERAGE",
        "subsections": [
            ("2.1 Covered Services",
             "This policy covers the following medical services when provided by an in-network "
             "provider: preventive care and annual wellness exams, primary care visits, "
             "specialist consultations, emergency room services, urgent care visits, "
             "hospitalization and inpatient surgery, outpatient surgery and procedures, "
             "laboratory tests and diagnostic imaging, prescription drugs listed on the "
             "formulary, mental health and substance use disorder treatment, physical therapy "
             "up to 30 sessions per calendar year, and maternity and newborn care."),
            ("2.2 Preventive Care",
             "Preventive care services are covered at 100% with no cost-sharing when provided "
             "by an in-network provider. Covered preventive services include annual physical "
             "exams, recommended immunizations, cancer screenings, blood pressure and "
             "cholesterol checks, and routine gynecological exams."),
            ("2.3 Emergency Services",
             "Emergency room services are covered regardless of whether the provider is "
             "in-network or out-of-network. The policyholder is responsible for the applicable "
             "emergency copayment listed in Section 4. If the policyholder is admitted to the "
             "hospital following an emergency room visit, the inpatient cost-sharing applies "
             "rather than the emergency copayment."),
            ("2.4 Prescription Drugs",
             "Prescription drugs are covered according to the four-tier formulary:\n"
             "  Tier 1 (Generic): $10 copay per 30-day supply\n"
             "  Tier 2 (Preferred Brand): $35 copay per 30-day supply\n"
             "  Tier 3 (Non-Preferred Brand): $70 copay per 30-day supply\n"
             "  Tier 4 (Specialty): 20% coinsurance after deductible\n\n"
             "Mail-order prescriptions for maintenance medications are available at 2.5 times "
             "the 30-day copay for a 90-day supply."),
        ],
    },
    {
        "number": "3",
        "title": "EXCLUSIONS",
        "subsections": [
            ("3.1 General Exclusions",
             "This policy does not cover: cosmetic procedures not medically necessary, "
             "experimental or investigational treatments, services received outside the United "
             "States except for emergency care, dental care except as stated in Section 2, "
             "vision care except for diabetic eye exams, custodial care or long-term care, "
             "weight loss programs or bariatric surgery unless medically necessary, and "
             "infertility treatments beyond initial diagnosis."),
            ("3.2 Pre-existing Conditions",
             "Pre-existing conditions are covered under this policy after a waiting period of "
             "12 months from the effective date of coverage, provided the policyholder "
             "maintains continuous coverage during that period. A pre-existing condition is "
             "defined as any condition for which the policyholder received diagnosis, treatment, "
             "or medical advice in the 12 months prior to the effective date."),
            ("3.3 Limitations on Mental Health Services",
             "Mental health and substance use disorder services are subject to the same "
             "cost-sharing as medical and surgical services. Inpatient psychiatric care is "
             "limited to 30 days per calendar year. Outpatient mental health visits are limited "
             "to 52 sessions per calendar year."),
        ],
    },
    {
        "number": "4",
        "title": "COST SHARING",
        "subsections": [
            ("4.1 Annual Deductible",
             "The annual deductible is $1,500 per individual and $3,000 per family. The "
             "deductible resets on January 1 of each calendar year. Preventive care services "
             "and Tier 1 prescription drugs do not apply toward the deductible."),
            ("4.2 Copayments and Coinsurance",
             "After the deductible is met, the following cost-sharing applies:\n"
             "  Primary care visit: $25 copay\n"
             "  Specialist visit: $50 copay\n"
             "  Emergency room: $250 copay (waived if admitted)\n"
             "  Urgent care: $75 copay\n"
             "  Outpatient surgery: 20% coinsurance\n"
             "  Inpatient hospitalization: $500 per admission"),
            ("4.3 Out-of-Pocket Maximum",
             "The annual out-of-pocket maximum is $7,000 per individual and $14,000 per "
             "family. Once this limit is reached, the plan covers 100% of covered services "
             "for the remainder of the calendar year. The out-of-pocket maximum includes "
             "deductibles, copayments, and coinsurance but does not include premiums or "
             "charges for non-covered services."),
        ],
    },
    {
        "number": "5",
        "title": "CLAIMS",
        "subsections": [
            ("5.1 Filing a Claim",
             "In-network providers submit claims directly to Blue Shield on behalf of the "
             "policyholder. For out-of-network services, the policyholder must submit a "
             "completed claim form along with an itemised bill within 180 days of the date "
             "of service. Claim forms are available at blueshield.com or by calling member "
             "services at 1-800-555-0100."),
            ("5.2 Claim Review and Payment",
             "Claims are typically processed within 30 days of receipt. If additional "
             "information is needed, Blue Shield will notify the policyholder within 15 days. "
             "Payment for approved claims is made directly to the provider for in-network "
             "services, or to the policyholder for out-of-network reimbursements."),
            ("5.3 Disputes and Appeals",
             "If a claim is denied, the policyholder has the right to appeal within 180 days "
             "of receiving the denial notice. Appeals must be submitted in writing and include "
             "a statement of the reasons for the appeal along with any supporting medical "
             "documentation. Blue Shield will respond to appeals within 60 days. If the appeal "
             "is denied, the policyholder may request an independent external review."),
        ],
    },
    {
        "number": "6",
        "title": "TERMINATION",
        "subsections": [
            ("6.1 Voluntary Termination",
             "The policyholder may cancel coverage at any time by providing 30 days written "
             "notice to Blue Shield. Coverage continues through the end of the calendar month "
             "in which the notice is received. Premiums paid beyond the termination date will "
             "be refunded within 30 days."),
            ("6.2 Involuntary Termination",
             "Blue Shield may terminate coverage for any of the following reasons: non-payment "
             "of premiums after a 30-day grace period, material misrepresentation on the "
             "application, loss of eligibility as defined in Section 1, or fraud or intentional "
             "deception in obtaining benefits. Written notice will be provided at least 30 days "
             "before termination except in cases of fraud."),
            ("6.3 Continuation of Coverage",
             "Upon termination of group coverage, eligible individuals may continue coverage "
             "under COBRA for up to 18 months. COBRA election must be made within 60 days of "
             "receiving the COBRA notice. The policyholder is responsible for the full premium "
             "plus a 2% administrative fee during the COBRA period."),
        ],
    },
    {
        "number": "7",
        "title": "CONTACT INFORMATION",
        "subsections": [
            ("Contact Details",
             "Member Services:     1-800-555-0100 (Monday to Friday, 8am to 8pm EST)\n"
             "Claims Department:   claims@blueshield.com\n"
             "Provider Directory:  blueshield.com/providers\n"
             "Pharmacy Benefits:   1-800-555-0199\n"
             "Mental Health Line:  1-800-555-0177 (24 hours, 7 days a week)\n\n"
             "Blue Shield Health Insurance Company\n"
             "1234 Insurance Plaza, Suite 500\n"
             "Hartford, CT 06103"),
        ],
    },
]


def build_pdf(output_path: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 12, "BLUE SHIELD HEALTH INSURANCE POLICY", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Policy Number: BSH-2024-004521", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Effective Date: January 1, 2024", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Policyholder: Jane Doe", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(
        0, 6,
        "This document sets out the terms and conditions of your health insurance coverage "
        "with Blue Shield Health Insurance Company. Please read it carefully and keep it in "
        "a safe place. Contact member services if you have any questions about your benefits.",
        align="C",
    )

    # Sections
    for section in SECTIONS:
        pdf.add_page()
        # Section heading
        pdf.set_font("Helvetica", "B", 14)
        heading = f"{section['number']}. {section['title']}"
        pdf.cell(0, 10, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(100, 100, 100)
        pdf.set_line_width(0.4)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
        pdf.ln(4)

        for sub_title, sub_body in section["subsections"]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, sub_title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, sub_body)
            pdf.ln(4)

    pdf.output(output_path)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_policy.pdf"
    build_pdf(str(out))
