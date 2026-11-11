"""
Export Clean Resubmission Manuscript to Microsoft Word (.docx) format.
Generates paper/manuscript_clean_submission.docx formatted cleanly for Academia Quantum.
Embeds high-resolution publication figures and creates styled tables.
"""

import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_shading(cell, color_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="666666"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def clean_inline_markdown(text):
    # Remove markdown bold/italics markers for clean display or parse runs
    return text

def add_formatted_paragraph(doc, text, style='Normal', space_after=4, line_spacing=1.15, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    
    # Handle bold prefix if any
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|\$.*?\$)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = p.add_run(part[2:-2])
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
        elif part.startswith('*') and part.endswith('*') and not part.startswith('**'):
            run = p.add_run(part[1:-1])
            run.italic = True
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
        elif part.startswith('$') and part.endswith('$'):
            # Math notation
            run = p.add_run(part[1:-1])
            run.italic = True
            run.font.name = "Cambria Math"
            run.font.size = Pt(10.5)
        else:
            run = p.add_run(part)
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
    return p

def export_clean_manuscript(md_path, docx_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return
        
    doc = Document()
    
    # Page Margins: Standard 1.0 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header
        header = section.header
        hp = header.paragraphs[0]
        hp.text = "Academia Quantum\tResearch Article"
        hp.paragraph_format.space_after = Pt(12)
        if hp.runs:
            hp.runs[0].font.name = "Calibri"
            hp.runs[0].font.size = Pt(8.5)
            hp.runs[0].font.color.rgb = RGBColor(120, 120, 120)
            
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_table = False
    table_lines = []
    
    def process_table(t_lines):
        if not t_lines or len(t_lines) < 2:
            return
            
        parsed_rows = []
        for line in t_lines:
            line_str = line.strip()
            if not line_str.startswith("|"):
                continue
            cells = [c.strip() for c in line_str.split("|")[1:-1]]
            if all(set(c).issubset({"-", ":", " "}) for c in cells):
                continue
            parsed_rows.append(cells)
            
        if not parsed_rows:
            return
            
        n_cols = max(len(r) for r in parsed_rows)
        table = doc.add_table(rows=len(parsed_rows), cols=n_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(table)
        
        for r_idx, row in enumerate(parsed_rows):
            for c_idx, cell_text in enumerate(row):
                if c_idx < n_cols:
                    cell = table.cell(r_idx, c_idx)
                    cell.text = "" # clear default
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.line_spacing = 1.05
                    
                    # Clean text
                    clean_c = cell_text.replace("**", "").replace("*", "")
                    run = p.add_run(clean_c)
                    run.font.name = "Calibri"
                    
                    if r_idx == 0:
                        run.bold = True
                        run.font.size = Pt(8.5)
                        set_cell_shading(cell, "EBF2FA")
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    else:
                        run.font.size = Pt(8.0)
                        if c_idx == 0:
                            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        else:
                            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        
    for line in lines:
        line_strip = line.strip()
        
        # Check for figure embeds
        fig_match = re.search(r'\[See results/figures/(figure\d+.*?\.png)', line_strip)
        if fig_match:
            fig_filename = fig_match.group(1)
            fig_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures", fig_filename)
            if os.path.exists(fig_path):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(4)
                run_img = p_img.add_run()
                run_img.add_picture(fig_path, width=Inches(5.8))
            continue
            
        if line_strip.startswith("|"):
            in_table = True
            table_lines.append(line_strip)
            continue
        else:
            if in_table:
                process_table(table_lines)
                in_table = False
                table_lines = []
                
        if not line_strip:
            continue
            
        if line_strip.startswith("# "):
            title_text = line_strip[2:]
            h = doc.add_heading(level=0)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(8)
            run = h.add_run(title_text)
            run.font.name = "Calibri"
            run.font.size = Pt(17)
            run.bold = True
            run.font.color.rgb = RGBColor(17, 34, 68)
        elif line_strip.startswith("## "):
            h_text = line_strip[3:]
            h = doc.add_heading(level=1)
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(4)
            run = h.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(13)
            run.bold = True
            run.font.color.rgb = RGBColor(24, 43, 73)
        elif line_strip.startswith("### "):
            h_text = line_strip[4:]
            h = doc.add_heading(level=2)
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(3)
            run = h.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(11.5)
            run.bold = True
            run.font.color.rgb = RGBColor(34, 60, 95)
        elif line_strip.startswith("#### "):
            h_text = line_strip[5:]
            h = doc.add_heading(level=3)
            h.paragraph_format.space_before = Pt(8)
            h.paragraph_format.space_after = Pt(2)
            run = h.add_run(h_text)
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            run.bold = True
            run.italic = True
        elif line_strip.startswith("```"):
            continue
        elif line_strip.startswith("---"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
        elif line_strip.startswith("$$") and line_strip.endswith("$$"):
            # Display math
            eq_text = line_strip[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(eq_text)
            run.font.name = "Cambria Math"
            run.font.size = Pt(10.5)
            run.italic = True
        elif line_strip.startswith("*Table ") or line_strip.startswith("*Figure "):
            # Caption
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(line_strip.replace("*", ""))
            run.font.name = "Calibri"
            run.font.size = Pt(9.0)
            run.italic = True
            run.font.color.rgb = RGBColor(80, 80, 80)
        else:
            add_formatted_paragraph(doc, line_strip, space_after=4, line_spacing=1.15)
            
    if in_table:
        process_table(table_lines)
        
    doc.save(docx_path)
    print(f"Successfully generated clean submission Word document: {docx_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    md_in = os.path.join(base_dir, "paper", "manuscript_clean_submission.md")
    docx_out = os.path.join(base_dir, "paper", "manuscript_clean_submission.docx")
    export_clean_manuscript(md_in, docx_out)

