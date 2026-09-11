r"""
Exports the complete Amazon Support AI Agent project and generates:
1. Full project copy in C:/Users/sanja/Documents/amazon-support-ai-agent
2. Formatted Microsoft Word report: REPORT.docx
3. Formatted PDF report: REPORT.pdf
4. Styled HTML report: REPORT.html
5. Markdown report: REPORT.md
directly accessible in the user's Documents folder.
"""
import os
import shutil
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def copy_project_to_documents():
    src_dir = r"C:\Users\sanja\.gemini\antigravity\scratch\amazon-support-ai-agent"
    dest_dir = r"C:\Users\sanja\Documents\amazon-support-ai-agent"

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(
        src_dir,
        dest_dir,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
    )
    print(f"Successfully copied project to: {dest_dir}")
    return dest_dir


def create_docx_report(md_path: str, docx_path: str):
    doc = Document()

    # Set page margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    title = doc.add_heading("@AmazonHelp Customer Support AI Agent", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Engineering & Evaluation Research Report | Production Benchmark Submission")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.italic = True
    sub.runs[0].font.color.rgb = RGBColor(100, 100, 100)

    meta_p = doc.add_paragraph("Author / Candidate Submission | Target: anurag@hiverhq.com | Date: September 2026")
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_p.runs[0].font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    table_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# ") and "Engineering & Evaluation Report" in stripped:
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            process_markdown_table(doc, table_lines)
            in_table = False
            table_lines = []

        if not stripped:
            continue

        if stripped.startswith("## "):
            h = doc.add_heading(stripped[3:], level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(4)
        elif stripped.startswith("### "):
            h = doc.add_heading(stripped[4:], level=2)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(2)
        elif stripped.startswith("#### "):
            h = doc.add_heading(stripped[5:], level=3)
            h.paragraph_format.space_before = Pt(6)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style='List Bullet')
            _format_rich_text(p, stripped[2:])
        elif re.match(r'^\d+\.\s', stripped):
            p = doc.add_paragraph(style='List Number')
            content = re.sub(r'^\d+\.\s', '', stripped)
            _format_rich_text(p, content)
        elif stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            r = p.add_run(stripped.lstrip("> ").strip())
            r.font.italic = True
            r.font.color.rgb = RGBColor(120, 60, 0)
        elif stripped.startswith("```"):
            continue
        else:
            p = doc.add_paragraph()
            _format_rich_text(p, stripped)

    if in_table:
        process_markdown_table(doc, table_lines)

    doc.save(docx_path)
    print(f"Successfully generated DOCX report at: {docx_path}")


def _format_rich_text(paragraph, text: str):
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            code_parts = re.split(r'(`.*?`)', part)
            for cp in code_parts:
                if cp.startswith("`") and cp.endswith("`"):
                    c_run = paragraph.add_run(cp[1:-1])
                    c_run.font.name = "Consolas"
                    c_run.font.size = Pt(9.5)
                    c_run.font.color.rgb = RGBColor(30, 41, 59)
                else:
                    paragraph.add_run(cp)


def process_markdown_table(doc, table_lines):
    if len(table_lines) < 2:
        return
    header_raw = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows_raw = []
    for line in table_lines[2:]:
        rows_raw.append([c.strip() for c in line.strip("|").split("|")])

    table = doc.add_table(rows=len(rows_raw) + 1, cols=len(header_raw))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(header_raw):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
        tcPr = hdr_cells[i]._element.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'F1F5F9')
        tcPr.append(shd)

    for r_idx, row in enumerate(rows_raw):
        row_cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            if c_idx < len(row_cells):
                row_cells[c_idx].text = val
                for p in row_cells[c_idx].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def create_pdf_report_reportlab(md_path: str, pdf_path: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#232f3e'),
        alignment=1,
        spaceAfter=6
    )
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#232f3e'),
        spaceBefore=14,
        spaceAfter=6
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    story = []

    # Title Banner
    story.append(Paragraph("@AmazonHelp Customer Support AI Agent", title_style))
    story.append(Paragraph("Engineering & Evaluation Research Report | 200 Golden Cases Benchmark<br/>Submission: anurag@hiverhq.com | September 2026", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#ff9900'), spaceBefore=0, spaceAfter=12))

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    table_lines = []

    def flush_table():
        if len(table_lines) < 2:
            return
        headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
        rows = []
        for l in table_lines[2:]:
            rows.append([Paragraph(c.strip(), body_style) for c in l.strip("|").split("|")])
        header_paras = [Paragraph(f"<b>{h}</b>", body_style) for h in headers]
        table_data = [header_paras] + rows
        t = Table(table_data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 8))

    for line in lines:
        s = line.strip()

        if s.startswith("# ") and "Engineering & Evaluation Report" in s:
            continue

        if s.startswith("|") and s.endswith("|"):
            in_table = True
            table_lines.append(s)
            continue
        elif in_table:
            flush_table()
            in_table = False
            table_lines = []

        if not s or s.startswith("```"):
            continue

        # Format markdown bold for reportlab
        html_s = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', s)
        html_s = re.sub(r'`(.*?)`', r'<font face="Courier" size="8">\1</font>', html_s)

        if s.startswith("## "):
            story.append(Paragraph(html_s.replace("## ", ""), h1_style))
        elif s.startswith("### "):
            story.append(Paragraph(html_s.replace("### ", ""), h2_style))
        elif s.startswith("- ") or s.startswith("* "):
            story.append(Paragraph("&bull; " + html_s[2:], bullet_style))
        elif re.match(r'^\d+\.\s', s):
            story.append(Paragraph(html_s, bullet_style))
        elif s.startswith(">"):
            story.append(Paragraph(f"<i>{html_s.lstrip('> ').strip()}</i>", bullet_style))
        else:
            story.append(Paragraph(html_s, body_style))

    if in_table:
        flush_table()

    doc.build(story)
    print(f"Successfully generated PDF report via ReportLab at: {pdf_path}")


def main():
    docs_root = os.path.expanduser("~/Documents")
    proj_in_docs = copy_project_to_documents()

    md_path = os.path.join(proj_in_docs, "REPORT.md")

    # Generate documents inside C:\Users\sanja\Documents\amazon-support-ai-agent
    docx_path = os.path.join(proj_in_docs, "REPORT.docx")
    pdf_path = os.path.join(proj_in_docs, "REPORT.pdf")

    create_docx_report(md_path, docx_path)
    create_pdf_report_reportlab(md_path, pdf_path)

    # Also place copies directly in C:\Users\sanja\Documents for instant 1-click access
    for fname in ["REPORT.docx", "REPORT.pdf", "REPORT.html", "REPORT.md", "README.md"]:
        src_f = os.path.join(proj_in_docs, fname)
        dst_f = os.path.join(docs_root, f"Amazon_Support_AI_Agent_{fname}")
        shutil.copy2(src_f, dst_f)
        print(f"Copied to Documents: {dst_f}")

    print("\nAll documents successfully saved in Documents in File Manager!")


if __name__ == "__main__":
    main()
