"""Generate a synthetic scanned PDF — image-only pages, no text layer.

Each page is rendered as a PIL image (simulating a scanner: slight rotation,
noise, off-white background) then embedded in a PDF via fpdf. pypdf will
extract zero text from these pages, triggering the OCR fallback path.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent

PAGES = [
    {
        "title": "MEETING MINUTES",
        "subtitle": "Board of Directors -- BlueTech Inc.",
        "date": "Date: March 14, 2026",
        "body": [
            "Attendees: J. Smith (Chair), A. Patel, R. Chen, M. Torres, L. Kim",
            "",
            "1. Call to Order",
            "   The meeting was called to order at 9:05 AM by Chair J. Smith.",
            "",
            "2. Approval of Previous Minutes",
            "   Minutes from February 2026 approved unanimously.",
            "",
            "3. Financial Report",
            "   CFO A. Patel presented Q1 financials. Revenue: $1.54M (+12% YoY).",
            "   Operating expenses: $980K. Net income: $560K. Cash reserves: $2.1M.",
            "",
            "4. Product Roadmap",
            "   CTO R. Chen outlined Q2 priorities: vector search improvements,",
            "   document ingestion pipeline, and MCP server integration.",
            "",
            "5. Action Items",
            "   - A. Patel: prepare Q2 budget proposal by March 28",
            "   - R. Chen: deliver pipeline v2 spec by March 21",
            "   - M. Torres: schedule customer advisory board for April",
        ],
    },
    {
        "title": "MEETING MINUTES (continued)",
        "subtitle": "Board of Directors -- BlueTech Inc.",
        "date": "Date: March 14, 2026  |  Page 2",
        "body": [
            "6. Legal Update",
            "   Counsel L. Kim confirmed IP filings for the hybrid search algorithm",
            "   are proceeding. Patent application number: US2026/0041872.",
            "",
            "7. Hiring Update",
            "   Three senior engineers hired in February. Two open roles remain:",
            "   Staff ML Engineer and Senior Backend Engineer.",
            "",
            "8. Strategic Partnerships",
            "   Letter of intent signed with DataVault Corp for OEM integration.",
            "   Deal value: $400K ARR. Expected close: Q2 2026.",
            "",
            "9. Next Meeting",
            "   Scheduled for April 11, 2026 at 9:00 AM.",
            "",
            "10. Adjournment",
            "    Meeting adjourned at 11:22 AM.",
            "",
            "Signed: ________________________",
            "        J. Smith, Chair",
        ],
    },
]


def _render_page(content: dict, width: int = 850, height: int = 1100) -> Image.Image:
    """Render a page dict to a PIL image simulating a scanned document."""
    rng = random.Random(42)

    # Off-white background with slight noise
    img = Image.new("RGB", (width, height), (252, 250, 248))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            noise = rng.randint(-4, 4)
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    draw = ImageDraw.Draw(img)

    # Try to use a system font, fall back to default
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        font_date = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font_title = font_sub = font_body = font_date = ImageFont.load_default()

    margin = 80
    y = 60

    # Title
    draw.text((width // 2, y), content["title"], font=font_title, fill=(20, 20, 20), anchor="mm")
    y += 40
    draw.text((width // 2, y), content["subtitle"], font=font_sub, fill=(60, 60, 60), anchor="mm")
    y += 24
    draw.text((width // 2, y), content["date"], font=font_date, fill=(80, 80, 80), anchor="mm")
    y += 20

    # Horizontal rule
    draw.line([(margin, y), (width - margin, y)], fill=(120, 120, 120), width=1)
    y += 20

    # Body lines
    for line in content["body"]:
        draw.text((margin, y), line, font=font_body, fill=(30, 30, 30))
        y += 22

    # Slight rotation to simulate scanner skew
    img = img.rotate(rng.uniform(-0.4, 0.4), fillcolor=(252, 250, 248), expand=False)

    # Light blur to simulate scanner softness
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img


def build_scanned_pdf():
    from fpdf import FPDF
    import io

    pdf = FPDF()

    for content in PAGES:
        img = _render_page(content)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        pdf.add_page()
        pdf.image(buf, x=0, y=0, w=210)  # full A4 width in mm

    out_path = OUT / "sample_scanned_minutes.pdf"
    pdf.output(str(out_path))
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    build_scanned_pdf()
