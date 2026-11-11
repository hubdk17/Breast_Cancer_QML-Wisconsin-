"""
Export Revised Marked Manuscript to Microsoft Word (.docx) format.
Generates paper/manuscript_marked_revision.docx formatted for Academia Quantum resubmission.
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def export_file_to_docx(md_path, docx_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} does not exist.")
        return
        
    doc = Document()
    
    # Page Margins: Standard 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
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
        table.style = "Table Grid"
        
        for r_idx, row in enumerate(parsed_rows):
            for c_idx, cell_text in enumerate(row):
                if c_idx < n_cols:
                    cell = table.cell(r_idx, c_idx)
                    cell.text = cell_text
                    if r_idx == 0:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.bold = True
                                run.font.size = Pt(9.5)
                    else:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = Pt(9.0)
        doc.add_paragraph()
        
    for line in lines:
        line_strip = line.strip()
        
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
            h = doc.add_heading(line_strip[2:], level=1)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif line_strip.startswith("## "):
            h = doc.add_heading(line_strip[3:], level=2)
        elif line_strip.startswith("### "):
            h = doc.add_heading(line_strip[4:], level=3)
        elif line_strip.startswith("#### "):
            h = doc.add_heading(line_strip[5:], level=4)
        elif line_strip.startswith("```"):
            continue
        elif line_strip.startswith("---"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
        else:
            p = doc.add_paragraph()
            clean_text = line_strip
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(clean_text)
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            
    if in_table:
        process_table(table_lines)
        
    doc.save(docx_path)
    print(f"Successfully exported Word document: {docx_path}")

def export_all():
    paper_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper")
    export_file_to_docx(os.path.join(paper_dir, "manuscript_marked_revision.md"),
                        os.path.join(paper_dir, "manuscript_marked_revision.docx"))
    export_file_to_docx(os.path.join(paper_dir, "response_to_reviewers.md"),
                        os.path.join(paper_dir, "response_to_reviewers.docx"))

if __name__ == "__main__":
    export_all()
