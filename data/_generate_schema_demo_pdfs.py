"""Generate sample PDFs used by examples/feature-demos/schema_inference_demo.py.

Produces three files alongside the existing sample_policy.pdf:
    data/sample_contract.pdf
    data/sample_invoice.pdf
    data/sample_research.pdf

Run once:
    python3 data/_generate_schema_demo_pdfs.py

Requires fpdf2: pip install fpdf2
"""
from __future__ import annotations

from pathlib import Path
from fpdf import FPDF


DATA_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    return pdf


def _heading(pdf: FPDF, text: str, size: int = 13) -> None:
    pdf.set_font("Helvetica", "B", size)
    pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(120, 120, 120)
    pdf.set_line_width(0.3)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)


def _body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, text)
    pdf.ln(4)


def _label_value(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(55, 6, label, new_x="RIGHT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")


# ---------------------------------------------------------------------------
# Contract PDF
# ---------------------------------------------------------------------------

def build_contract(path: str) -> None:
    pdf = _new_pdf()

    # Cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.ln(20)
    pdf.cell(0, 12, "SOFTWARE CONSULTING AGREEMENT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Between TechCorp Inc. and BuildRight Solutions LLC", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Effective Date: September 1, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Contract Value: $180,000 per annum", align="C", new_x="LMARGIN", new_y="NEXT")

    # Recitals
    pdf.add_page()
    _heading(pdf, "RECITALS")
    _body(pdf,
        "WHEREAS TechCorp Inc. (\"Company\") wishes to engage BuildRight Solutions LLC "
        "(\"Vendor\") for software consulting services; and\n\n"
        "WHEREAS the Vendor represents that it has the expertise and capacity to perform "
        "such services; and\n\n"
        "WHEREAS the parties wish to set forth the terms and conditions governing their "
        "working relationship;\n\n"
        "NOW, THEREFORE, in consideration of the mutual covenants herein, the parties agree "
        "as follows.")

    _heading(pdf, "1. OBLIGATIONS")
    _body(pdf,
        "1.1 Vendor Obligations. Vendor shall deliver software consulting services as "
        "described in each Statement of Work (SOW) executed under this Agreement. Each SOW "
        "shall specify deliverables, timelines, and acceptance criteria.\n\n"
        "1.2 Company Obligations. Company shall provide Vendor with timely access to systems, "
        "personnel, and documentation reasonably required to perform the services. Company "
        "shall review and accept or reject deliverables within 10 business days of submission.")

    _heading(pdf, "2. TERMINATION")
    _body(pdf,
        "2.1 Termination for Convenience. Either party may terminate this Agreement with "
        "30 days written notice to the other party. Vendor shall be compensated for all "
        "services performed and expenses incurred through the effective date of termination.\n\n"
        "2.2 Termination for Cause. Company may terminate this Agreement immediately and "
        "without notice if Vendor materially breaches any provision and fails to cure such "
        "breach within 10 business days of receiving written notice of the breach.")

    _heading(pdf, "3. GOVERNING LAW AND ARBITRATION")
    _body(pdf,
        "3.1 Governing Law. This Agreement shall be governed by and construed in accordance "
        "with the laws of the State of Delaware, without regard to its conflict-of-laws "
        "provisions.\n\n"
        "3.2 Arbitration. Any dispute, controversy, or claim arising out of or relating to "
        "this Agreement shall be resolved by binding arbitration administered by the American "
        "Arbitration Association under its Commercial Arbitration Rules. The arbitration "
        "shall take place in New York, New York.")

    pdf.add_page()
    _heading(pdf, "4. INDEMNIFICATION AND LIABILITY")
    _body(pdf,
        "4.1 Indemnification. Each party (the \"Indemnifying Party\") shall indemnify, "
        "defend, and hold harmless the other party and its officers, directors, employees, "
        "and agents (collectively, \"Indemnitees\") from and against any third-party claims, "
        "damages, losses, and expenses, including reasonable attorneys' fees, arising out of "
        "or resulting from the Indemnifying Party's breach of this Agreement or negligence.\n\n"
        "4.2 Limitation of Liability. In no event shall either party's liability to the "
        "other exceed the total fees paid or payable under this Agreement in the 12 months "
        "preceding the event giving rise to the claim. Neither party shall be liable for "
        "indirect, incidental, consequential, or punitive damages.")

    _heading(pdf, "5. CONFIDENTIALITY AND INTELLECTUAL PROPERTY")
    _body(pdf,
        "5.1 Confidentiality. Vendor shall treat all Company information as strictly "
        "confidential and shall not disclose it to any third party without prior written "
        "consent. This obligation survives termination for a period of 5 years.\n\n"
        "5.2 Intellectual Property. All work product, deliverables, and inventions created "
        "by Vendor under this Agreement are works-for-hire and the exclusive property of "
        "Company. Vendor hereby assigns all intellectual property rights therein to Company.\n\n"
        "5.3 Warranties. Vendor warrants that the services will be performed in a "
        "professional and workmanlike manner consistent with industry standards, and that "
        "deliverables will not infringe any third-party intellectual property rights.")

    _heading(pdf, "6. FORCE MAJEURE AND GENERAL PROVISIONS")
    _body(pdf,
        "6.1 Force Majeure. Neither party shall be liable for any delay or failure to "
        "perform its obligations due to circumstances beyond its reasonable control, "
        "including acts of God, government action, natural disasters, or pandemic events.\n\n"
        "6.2 Entire Agreement. This Agreement, together with all executed SOWs, constitutes "
        "the entire agreement between the parties and supersedes all prior representations, "
        "understandings, and agreements relating to its subject matter.\n\n"
        "6.3 Severability. If any provision of this Agreement is found to be unenforceable, "
        "the remaining provisions shall continue in full force and effect.")

    pdf.output(path)
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Invoice PDF
# ---------------------------------------------------------------------------

def build_invoice(path: str) -> None:
    pdf = _new_pdf()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "INVOICE", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    _label_value(pdf, "Invoice Number:", "INV-2026-0891")
    _label_value(pdf, "Invoice Date:",   "August 1, 2026")
    _label_value(pdf, "Due Date:",       "August 31, 2026")
    _label_value(pdf, "Purchase Order:", "PO-ACME-2026-214")
    pdf.ln(6)

    # Bill To / Vendor
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(85, 7, "Bill To:", new_x="RIGHT")
    pdf.cell(0, 7, "Vendor:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    bill_to = ["Acme Manufacturing Corp", "200 Industrial Parkway",
               "Detroit, MI 48201", "billing@acme.example.com"]
    vendor  = ["Blue Solutions Inc.", "123 Innovation Drive",
               "San Francisco, CA 94105", "invoices@bluesolutions.example.com"]
    for b, v in zip(bill_to, vendor):
        pdf.cell(85, 6, b, new_x="RIGHT")
        pdf.cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Line items table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(220, 220, 220)
    cols = [90, 25, 30, 25]
    headers = ["Description", "Qty", "Unit Price", "Total"]
    for col_w, h in zip(cols, headers):
        pdf.cell(col_w, 8, h, border=1, fill=True, new_x="RIGHT")
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    line_items = [
        ("Cloud Infrastructure Design & Architecture",  "1",  "$12,000", "$12,000"),
        ("Security Audit (40 hours @ $250/hr)",         "40", "$250",    "$10,000"),
        ("Staff Training Workshop (full day, 3 groups)","3",  "$800",    " $2,400"),
        ("Data Migration Planning & Documentation",     "1",  "$4,500",  " $4,500"),
        ("Project Management (20 hours @ $175/hr)",     "20", "$175",    " $3,500"),
    ]
    for desc, qty, unit, total in line_items:
        pdf.cell(cols[0], 7, desc, border=1, new_x="RIGHT")
        pdf.cell(cols[1], 7, qty,  border=1, align="C", new_x="RIGHT")
        pdf.cell(cols[2], 7, unit, border=1, align="R", new_x="RIGHT")
        pdf.cell(cols[3], 7, total,border=1, align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    # Totals
    for label, value in [("Subtotal:", "$32,400"), ("Tax (8.5%):", "$2,754"),
                         ("Total Due:", "$35,154")]:
        pdf.set_font("Helvetica", "B" if label == "Total Due:" else "", 10)
        pdf.cell(130, 7, label, align="R", new_x="RIGHT")
        pdf.cell(40, 7, value, align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    _heading(pdf, "PAYMENT INSTRUCTIONS")
    _body(pdf,
        "Payment Terms: Net-30 from invoice date.\n\n"
        "ACH Transfer:\n"
        "  Bank: First National Bank\n"
        "  Routing Number: 121000358\n"
        "  Account Number: 9876543210\n"
        "  Account Name: Blue Solutions Inc.\n\n"
        "Please include the invoice number INV-2026-0891 in the payment reference.\n"
        "For billing questions contact: invoices@bluesolutions.example.com or 1-800-555-0200.\n\n"
        "Remit to: Blue Solutions Inc., 123 Innovation Drive, San Francisco, CA 94105")

    pdf.output(path)
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Research paper PDF
# ---------------------------------------------------------------------------

def build_research(path: str) -> None:
    pdf = _new_pdf()

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.ln(15)
    pdf.multi_cell(0, 10,
        "HybridRank: Combining Sparse and Dense Retrieval\n"
        "for Open-Domain Question Answering", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(0, 7, "Alexandra Chen, Marcus Williams, Priya Nair", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Blue AI Research Lab", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "{achen, mwilliams, pnair}@blueailab.example.com", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    _heading(pdf, "Abstract")
    _body(pdf,
        "We present HybridRank, a retrieval architecture that combines sparse BM25 retrieval "
        "with dense neural embeddings for open-domain question answering. Our hypothesis is "
        "that neither approach alone achieves optimal recall-precision trade-offs on "
        "heterogeneous corpora. We evaluate on three standard benchmarks -- Natural Questions, "
        "TriviaQA, and WebQuestions -- and demonstrate a 38% improvement in MRR@10 over BM25 "
        "alone and a 12% improvement over dense-only retrieval. We propose a novel score "
        "fusion method that outperforms both linear interpolation and reciprocal rank fusion.\n\n"
        "Keywords: hybrid retrieval, BM25, dense embeddings, question answering, "
        "neural information retrieval, score fusion")

    _heading(pdf, "1. Introduction")
    _body(pdf,
        "Information retrieval has been transformed by the advent of transformer-based "
        "language models. While BM25 remains a strong keyword-based baseline with excellent "
        "recall, dense retrieval methods using bi-encoder architectures have demonstrated "
        "superior performance on several standard benchmarks for semantic matching.\n\n"
        "However, practitioners have observed that dense retrievers struggle with rare entity "
        "names and exact keyword matching, while BM25 fails on queries that require semantic "
        "understanding. This motivates a hybrid approach that combines the complementary "
        "strengths of both paradigms.\n\n"
        "In this paper we make the following contributions: (1) a two-stage retrieval "
        "pipeline combining BM25 candidate retrieval with neural reranking; (2) a novel "
        "Reciprocal Score Fusion (RSF) method that outperforms prior fusion techniques; "
        "and (3) a comprehensive evaluation on three open-domain QA benchmarks.")

    pdf.add_page()
    _heading(pdf, "2. Methodology")
    _body(pdf,
        "2.1 Stage 1 -- BM25 Retrieval. We use BM25+ with tuned parameters k1=1.6, b=0.75 "
        "to retrieve the top-100 candidate passages from the corpus. BM25 provides high "
        "recall (89.3% on NQ) at low computational cost and serves as an efficient first "
        "filter over large corpora.\n\n"
        "2.2 Stage 2 -- Neural Reranking. We apply a cross-encoder reranker fine-tuned on "
        "MS MARCO to score each candidate passage against the query. The cross-encoder "
        "jointly encodes the query and passage, enabling full attention across both. We "
        "select the top-10 passages by reranker score.\n\n"
        "2.3 Score Fusion. We propose Reciprocal Score Fusion (RSF) as an alternative to "
        "linear interpolation. RSF assigns each passage a fused score computed as the "
        "harmonic mean of its normalised BM25 rank and reranker score. This reduces "
        "sensitivity to score scale differences between the two retrievers.\n\n"
        "2.4 Training Details. The reranker is initialised from DeBERTa-v3-large and "
        "fine-tuned for 3 epochs on MS MARCO with a batch size of 32 and learning rate "
        "of 2e-5. Negative passages are sampled using BM25 hard negatives.")

    _heading(pdf, "3. Findings and Discussion")
    _body(pdf,
        "Table 1 reports MRR@10 and Recall@100 on three benchmarks. HybridRank achieves "
        "state-of-the-art results across all three, with the largest gains on Natural "
        "Questions where semantic understanding is most critical.\n\n"
        "Key findings:\n"
        "  * BM25+RSF outperforms BM25+linear by 4.2% MRR@10 on NQ.\n"
        "  * Dense-only retrieval degrades on rare entity queries (e.g. obscure "
        "    person names) -- BM25 recall supplements these gaps.\n"
        "  * The two-stage pipeline adds only 12ms per query vs. BM25-only at p99.\n\n"
        "Discussion. Failure modes include multi-hop questions requiring cross-passage "
        "reasoning and long documents where passage boundary effects degrade reranker "
        "performance. We discuss these limitations and propose future directions.")

    _heading(pdf, "4. Conclusion")
    _body(pdf,
        "HybridRank achieves state-of-the-art results on three open-domain QA benchmarks "
        "by combining the complementary strengths of BM25 and dense neural retrieval. "
        "Our RSF score fusion method is simple, parameter-free, and consistently "
        "outperforms linear interpolation.\n\n"
        "Future work includes: (1) extension to multilingual retrieval; (2) domain "
        "adaptation for specialised corpora (legal, biomedical); (3) integration with "
        "generative answer synthesis. Code and models are available at "
        "github.com/blueailab/hybridrank.")

    pdf.add_page()
    _heading(pdf, "References")
    refs = [
        "[1] Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA. "
        "EMNLP. DOI: 10.18653/v1/2020.emnlp-main.550",
        "[2] Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.",
        "[3] Nogueira & Cho (2019). Passage Re-ranking with BERT. arXiv:1901.04085",
        "[4] He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.",
        "[5] Nguyen et al. (2016). MS MARCO: A Human Generated Machine Reading Comprehension Dataset.",
        "[6] Kwiatkowski et al. (2019). Natural Questions: A Benchmark for QA Research. TACL.",
        "[7] Joshi et al. (2017). TriviaQA: A Reading Comprehension Dataset. ACL.",
    ]
    pdf.set_font("Helvetica", "", 9)
    for ref in refs:
        pdf.multi_cell(0, 6, ref)
        pdf.ln(2)

    pdf.output(path)
    print(f"Written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_contract(str(DATA_DIR / "sample_contract.pdf"))
    build_invoice(str(DATA_DIR / "sample_invoice.pdf"))
    build_research(str(DATA_DIR / "sample_research.pdf"))
    print("Done -- all three PDFs written to data/")
