"""Generate realistic test PDFs covering tables, embedded images, and mixed content."""
from pathlib import Path
import io

OUT = Path(__file__).parent


def _bar_chart_image(labels, values, width=160, height=80):
    """Return a PIL Image of a simple bar chart."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    max_val = max(values) or 1
    bar_w = (width - 20) // len(values)
    for i, (label, val) in enumerate(zip(labels, values)):
        x0 = 10 + i * bar_w + 2
        bar_h = int((val / max_val) * (height - 20))
        y0 = height - 10 - bar_h
        draw.rectangle([x0, y0, x0 + bar_w - 4, height - 10], fill="#4472C4")
        draw.text((x0, height - 9), label[:3], fill="black")

    return img


def build_table_invoice():
    """Invoice PDF with a rendered table of line items and a subtotal row."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "INVOICE", ln=True, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Invoice No: INV-2026-0042", ln=True)
    pdf.cell(0, 6, "Date: August 15, 2026", ln=True)
    pdf.cell(0, 6, "Bill To: Acme Corp, 123 Main St, Springfield", ln=True)
    pdf.cell(0, 6, "From:   BlueTech Services LLC", ln=True)
    pdf.ln(4)

    # Table header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(200, 220, 255)
    col_w = [80, 25, 35, 40]
    headers = ["Description", "Qty", "Unit Price", "Total"]
    for h, w in zip(headers, col_w):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    # Table rows
    rows = [
        ("Data pipeline consulting", "10", "$250.00", "$2,500.00"),
        ("Vector store setup", "1", "$800.00", "$800.00"),
        ("Document ingestion module", "1", "$1,200.00", "$1,200.00"),
        ("API integration (hours)", "5", "$150.00", "$750.00"),
        ("Testing and QA", "8", "$120.00", "$960.00"),
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(255, 255, 255)
    for row in rows:
        for val, w in zip(row, col_w):
            pdf.cell(w, 7, val, border=1)
        pdf.ln()

    # Subtotal rows
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(235, 235, 235)
    pdf.cell(col_w[0] + col_w[1] + col_w[2], 7, "Subtotal", border=1, fill=True)
    pdf.cell(col_w[3], 7, "$6,210.00", border=1, fill=True)
    pdf.ln()
    pdf.cell(col_w[0] + col_w[1] + col_w[2], 7, "Tax (8%)", border=1, fill=True)
    pdf.cell(col_w[3], 7, "$496.80", border=1, fill=True)
    pdf.ln()
    pdf.cell(col_w[0] + col_w[1] + col_w[2], 7, "TOTAL DUE", border=1, fill=True)
    pdf.cell(col_w[3], 7, "$6,706.80", border=1, fill=True)
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Payment due within 30 days. Bank transfer or credit card accepted.", ln=True)
    pdf.cell(0, 6, "Late payments subject to 1.5% monthly interest.", ln=True)

    pdf.output(str(OUT / "sample_table_invoice.pdf"))
    print("wrote sample_table_invoice.pdf")


def build_mixed_content():
    """PDF with text, an embedded bar chart image, and a summary table on the same page."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "Q2 2026 Performance Report", ln=True, align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "This report summarises the key performance indicators for Q2 2026. "
        "Revenue grew 18% year-over-year driven by strong demand in the enterprise "
        "segment. Operating costs remained flat despite headcount additions. "
        "The following chart shows monthly revenue for the quarter."
    )
    pdf.ln(3)

    # Embed bar chart image
    chart = _bar_chart_image(["Apr", "May", "Jun"], [420, 510, 610])
    buf = io.BytesIO()
    chart.save(buf, format="PNG")
    buf.seek(0)
    pdf.image(buf, x=25, w=80, h=40)
    pdf.ln(44)

    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Revenue peaked in June at $610k, up from $420k in April. "
        "Customer acquisition cost decreased to $142 per customer, a 12% improvement "
        "over Q1. Churn rate held steady at 2.1% monthly."
    )
    pdf.ln(4)

    # Summary table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Key Metrics Summary", ln=True)
    pdf.ln(1)

    pdf.set_fill_color(200, 220, 255)
    for header, w in [("Metric", 90), ("Q1 2026", 40), ("Q2 2026", 40), ("Change", 20)]:
        pdf.cell(w, 7, header, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    metrics = [
        ("Total Revenue", "$1.38M", "$1.54M", "+12%"),
        ("Gross Margin", "61%", "64%", "+3pp"),
        ("New Customers", "312", "389", "+25%"),
        ("Churn Rate", "2.3%", "2.1%", "-0.2pp"),
        ("CAC", "$161", "$142", "-12%"),
    ]
    for row in metrics:
        for val, w in zip(row, [90, 40, 40, 20]):
            pdf.cell(w, 7, val, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5,
        "Note: Revenue figures are unaudited. CAC = Customer Acquisition Cost. "
        "pp = percentage points. All figures in USD unless stated otherwise."
    )

    pdf.output(str(OUT / "sample_mixed_content.pdf"))
    print("wrote sample_mixed_content.pdf")


def build_multipage_with_tables():
    """Multi-page PDF: cover, text-heavy page, table page, and a references page."""
    from fpdf import FPDF

    pdf = FPDF()

    # Page 1 — cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(30)
    pdf.cell(0, 12, "Technical Evaluation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Vector Database Benchmarking Study", ln=True, align="C")
    pdf.cell(0, 8, "August 2026 | BlueTech Research", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Prepared by: Engineering Team", ln=True, align="C")
    pdf.cell(0, 6, "Version: 1.0 DRAFT", ln=True, align="C")

    # Page 2 — abstract + body text
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Abstract", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "This report evaluates three vector database backends (Redis, ChromaDB, Weaviate) "
        "across five dimensions: indexing throughput, query latency, recall accuracy, "
        "memory footprint, and operational complexity. Benchmarks were conducted on a "
        "corpus of 500,000 document chunks derived from publicly available research papers. "
        "Results indicate that Redis Stack achieves the best latency profile while "
        "ChromaDB offers the simplest operational model for development environments."
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Introduction", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Retrieval-Augmented Generation (RAG) pipelines depend critically on the "
        "performance of the underlying vector store. As corpus sizes grow into the "
        "millions of chunks, the choice of vector database has significant implications "
        "for both system latency and infrastructure cost. This study provides empirical "
        "data to guide that decision in production deployments."
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Methodology", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "All benchmarks were run on identical hardware: 16-core AMD EPYC, 64GB RAM, "
        "NVMe SSD. Each backend was configured with default settings except for index "
        "parameters which were tuned to the ef_construction=200, M=16 HNSW baseline. "
        "Queries were drawn from a held-out set of 10,000 natural language questions. "
        "Recall@10 was computed against a brute-force exact search ground truth."
    )

    # Page 3 — results table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. Results", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Table 1 summarises the benchmark results. Latency is reported as p50/p99 in "
        "milliseconds. Throughput is vectors indexed per second. Recall is Recall@10."
    )
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(200, 220, 255)
    cols = [("Backend", 38), ("Index Throughput", 38), ("p50 Latency", 32),
            ("p99 Latency", 32), ("Recall@10", 30), ("RAM (GB)", 22)]
    for h, w in cols:
        pdf.cell(w, 7, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    data = [
        ("Redis Stack", "48,200 v/s", "3.1 ms", "9.4 ms", "0.962", "8.2"),
        ("ChromaDB", "22,100 v/s", "7.8 ms", "24.1 ms", "0.971", "4.1"),
        ("Weaviate", "39,800 v/s", "4.2 ms", "12.8 ms", "0.958", "11.4"),
    ]
    for row in data:
        for val, (_, w) in zip(row, cols):
            pdf.cell(w, 7, val, border=1)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Conclusion", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Redis Stack is recommended for latency-sensitive production workloads. "
        "ChromaDB is suitable for development and small-scale deployments due to its "
        "zero-infrastructure footprint. Weaviate offers a middle ground with strong "
        "recall and reasonable operational overhead. Future work will extend these "
        "benchmarks to sparse-dense hybrid retrieval configurations."
    )

    # Page 4 — references
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "References", ln=True)
    pdf.set_font("Helvetica", "", 9)
    refs = [
        "[1] Johnson et al. (2021). Billion-scale similarity search with GPUs. IEEE TPAMI.",
        "[2] Malkov & Yashunin (2020). Efficient and robust approximate nearest neighbor search "
        "using HNSW. IEEE TPAMI.",
        "[3] Lewis et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. "
        "NeurIPS 2020.",
        "[4] Redis Ltd. (2024). RediSearch vector similarity documentation.",
        "[5] ChromaDB (2024). Chroma embedding database documentation. GitHub.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, ref)
        pdf.ln(1)

    pdf.output(str(OUT / "sample_multipage_tables.pdf"))
    print("wrote sample_multipage_tables.pdf")


if __name__ == "__main__":
    build_table_invoice()
    build_mixed_content()
    build_multipage_with_tables()
    print("done")
