"""
Export clean manuscript directly to publication-ready PDF using ReportLab.
Features:
- Two-pass NumberedCanvas for running headers and 'Page X of Y' footers
- Styled academic headings and structured abstract callout
- Auto-wrapped, cleanly bordered tables with shaded headers
- High-resolution embedded figures with captions
"""

import os
import re
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Draw header on all pages except first
        if self._pageNumber > 1:
            self.drawString(54, 752, "Academia Quantum  |  Research Article")
            self.drawRightString(612 - 54, 752, "Kaila et al.  |  Circuit Topologies & Feature Allocation for QIVCs")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.6)
            self.line(54, 746, 612 - 54, 746)
            
        # Draw footer on all pages
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.6)
        self.line(54, 46, 612 - 54, 46)
        self.drawString(54, 34, "GitHub: https://github.com/hubdk17/Breast_Cancer_QML-Wisconsin-.git")
        self.drawRightString(612 - 54, 34, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def clean_text_for_reportlab(text):
    # Escape xml entities
    t = text.replace("&", "&amp;")
    # Convert markdown bold/italics to html tags
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*', r'<i>\1</i>', t)
    # Clean latex math delimiters for display
    t = re.sub(r'\$(.*?)\$', r'<i>\1</i>', t)
    return t

def generate_pdf(md_path, pdf_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return False
        
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#112244"),
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
        spaceAfter=6
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceAfter=3
    )
    
    affil_style = ParagraphStyle(
        'AffilStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#444444"),
        spaceAfter=2
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#182B49"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=colors.HexColor("#243D61"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'SectionH3',
        parent=styles['Heading3'],
        fontName='Helvetica-BoldOblique',
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#334E73"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.0,
        leading=12.5,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=5
    )
    
    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#112244"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1A1A1A"),
        alignment=1 # Center
    )
    
    table_cell_left_style = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1A1A1A"),
        alignment=0 # Left
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F223D"),
        alignment=1
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#444444"),
        spaceBefore=3,
        spaceAfter=8,
        keepWithNext=False
    )
    
    story = []
    
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_table = False
    table_lines = []
    
    def process_table_lines(t_lines):
        if not t_lines or len(t_lines) < 2:
            return
            
        parsed_rows = []
        for line in t_lines:
            ls = line.strip()
            if not ls.startswith("|"):
                continue
            cells = [c.strip() for c in ls.split("|")[1:-1]]
            if all(set(c).issubset({"-", ":", " "}) for c in cells):
                continue
            parsed_rows.append(cells)
            
        if not parsed_rows:
            return
            
        n_cols = max(len(r) for r in parsed_rows)
        # Calculate column widths to fit 504 pt available width
        if n_cols == 2:
            col_widths = [160, 344]
        elif n_cols <= 4:
            col_widths = [504 / n_cols] * n_cols
        elif n_cols <= 6:
            col_widths = [114] + [(390 / (n_cols - 1))] * (n_cols - 1)
        elif n_cols <= 8:
            col_widths = [84, 54] + [(366 / (n_cols - 2))] * (n_cols - 2)
        elif n_cols <= 10:
            col_widths = [114] + [(390 / (n_cols - 1))] * (n_cols - 1)
        else:
            col_widths = [504 / n_cols] * n_cols
            
        table_data = []
        for r_idx, row in enumerate(parsed_rows):
            row_data = []
            for c_idx, cell_text in enumerate(row):
                if c_idx < n_cols:
                    clean_cell = clean_text_for_reportlab(cell_text)
                    if r_idx == 0:
                        p = Paragraph(clean_cell, table_header_style)
                    else:
                        if c_idx == 0:
                            p = Paragraph(clean_cell, table_cell_left_style)
                        else:
                            p = Paragraph(clean_cell, table_cell_style)
                    row_data.append(p)
            while len(row_data) < n_cols:
                row_data.append(Paragraph("", table_cell_style))
            table_data.append(row_data)
            
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EBF2FA")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.HexColor("#99AABF")),
            ('LINEBELOW', (0, 0), (-1, 0), 0.8, colors.HexColor("#99AABF")),
            ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.HexColor("#99AABF")),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#E2E7ED")),
        ]
        # Alternating row background
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                t_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F9FBFC")))
        t.setStyle(TableStyle(t_style))
        story.append(Spacer(1, 4))
        story.append(t)
        story.append(Spacer(1, 3))

    for line in lines:
        line_strip = line.strip()
        
        # Check for figure embeds
        fig_match = re.search(r'\[See results/figures/(figure\d+.*?\.png)', line_strip)
        if fig_match:
            fig_filename = fig_match.group(1)
            fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures", fig_filename)
            if os.path.exists(fig_path):
                story.append(Spacer(1, 6))
                # Max width 504 pt, maintain aspect ratio
                img = Image(fig_path, width=490, height=210)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 3))
            continue
            
        if line_strip.startswith("|"):
            in_table = True
            table_lines.append(line_strip)
            continue
        else:
            if in_table:
                process_table_lines(table_lines)
                in_table = False
                table_lines = []
                
        if not line_strip:
            continue
            
        if line_strip.startswith("# "):
            title_text = clean_text_for_reportlab(line_strip[2:])
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 2))
        elif line_strip.startswith("**Journal:**"):
            meta_text = clean_text_for_reportlab(line_strip)
            story.append(Paragraph(meta_text, meta_style))
        elif line_strip.startswith("**Authors:**"):
            story.append(Spacer(1, 4))
            author_text = clean_text_for_reportlab(line_strip)
            story.append(Paragraph(author_text, author_style))
        elif line_strip.startswith("$^1$") or line_strip.startswith("$^2$") or line_strip.startswith("$^*$"):
            affil_text = clean_text_for_reportlab(line_strip)
            story.append(Paragraph(affil_text, affil_style))
        elif line_strip.startswith("### Abstract"):
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#003366"), spaceBefore=4, spaceAfter=6))
            story.append(Paragraph("Abstract", abstract_heading_style))
        elif line_strip.startswith("## "):
            h_text = clean_text_for_reportlab(line_strip[3:])
            story.append(Paragraph(h_text, h1_style))
        elif line_strip.startswith("### "):
            h_text = clean_text_for_reportlab(line_strip[4:])
            story.append(Paragraph(h_text, h2_style))
        elif line_strip.startswith("#### "):
            h_text = clean_text_for_reportlab(line_strip[5:])
            story.append(Paragraph(h_text, h3_style))
        elif line_strip.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#D0D7DE"), spaceBefore=4, spaceAfter=4))
        elif line_strip.startswith("$$") and line_strip.endswith("$$"):
            eq_text = clean_text_for_reportlab(line_strip[2:-2].strip())
            eq_style = ParagraphStyle('DisplayEq', parent=body_style, alignment=1, fontName='Helvetica-Oblique', textColor=colors.HexColor("#0D1A30"))
            story.append(Spacer(1, 2))
            story.append(Paragraph(eq_text, eq_style))
            story.append(Spacer(1, 2))
        elif line_strip.startswith("*Table ") or line_strip.startswith("*Figure "):
            caption_text = clean_text_for_reportlab(line_strip)
            story.append(Paragraph(caption_text, caption_style))
        else:
            p_text = clean_text_for_reportlab(line_strip)
            story.append(Paragraph(p_text, body_style))
            
    if in_table:
        process_table_lines(table_lines)
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated clean publication PDF: {pdf_path}")
    print(f"PDF file size: {os.path.getsize(pdf_path)} bytes")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    md_in = os.path.join(base_dir, "paper", "manuscript_clean_submission.md")
    pdf_out = os.path.join(base_dir, "paper", "manuscript_clean_submission.pdf")
    generate_pdf(md_in, pdf_out)

