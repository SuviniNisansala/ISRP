"""Small python-docx helpers. The draft is built on the student's own template
(Final template.docx), so Heading 1/2/3, Normal (Times New Roman 12) and Caption styles match the
thesis exactly when the text is copied across."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

TEMPLATE = Path(__file__).resolve().parents[2] / "new-dataset-and-documens" / "Final template.docx"
SOURCE_DEFAULT = "Source: Compiled by author based on CMC (2025) assessment records and OpenStreetMap data"


class Doc:
    def __init__(self, template=TEMPLATE):
        if Path(template).exists():
            self.d = Document(template)
            body = self.d.element.body
            for el in list(body):
                if el.tag != qn("w:sectPr"):
                    body.remove(el)
        else:
            self.d = Document()
            st = self.d.styles["Normal"]
            st.font.name, st.font.size = "Times New Roman", Pt(12)
        self.fig_no = 0
        self.tab_no = 0
        self.chapter = 4
        self.width = Cm(14.4)

    # -- text ----------------------------------------------------------------------------
    def h(self, text, level):
        return self.d.add_heading(text, level=level)

    def p(self, text, italic=False, bold=False, size=None, color=None, align=None):
        para = self.d.add_paragraph()
        _runs_with_bold(para, text, italic=italic, bold=bold, size=size, color=color)
        para.paragraph_format.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(6)
        return para

    def bullets(self, items):
        for it in items:
            try:
                para = self.d.add_paragraph(style="List Paragraph")
            except KeyError:
                para = self.d.add_paragraph()
            _runs_with_bold(para, "•  " + it)
            para.paragraph_format.left_indent = Cm(0.8)
            para.paragraph_format.first_line_indent = Cm(-0.5)
            para.paragraph_format.space_after = Pt(3)

    def note(self, text):
        """Grey italic note for the author (to be deleted before submission)."""
        return self.p("[Author note: " + text + "]", italic=True, size=10, color=RGBColor(0x80, 0x80, 0x80))

    # -- captions --------------------------------------------------------------------------
    def _caption(self, text):
        try:
            para = self.d.add_paragraph(style="Caption")
        except KeyError:
            para = self.d.add_paragraph()
        r = para.add_run(text)
        r.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(0, 0, 0)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(2)
        return para

    def _source(self, text):
        para = self.d.add_paragraph()
        r = para.add_run(text)
        r.italic = True
        r.font.size = Pt(9.5)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(10)

    def next_fig(self):
        return f"Figure {self.chapter}.{self.fig_no + 1}"

    def next_tab(self):
        return f"Table {self.chapter}.{self.tab_no + 1}"

    def figure(self, path, caption, source=SOURCE_DEFAULT, width=None):
        self.fig_no += 1
        para = self.d.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(path), width=width or self.width)
        self._caption(f"Figure {self.chapter}.{self.fig_no}: {caption}")
        self._source(source)
        return f"Figure {self.chapter}.{self.fig_no}"

    def table(self, df: pd.DataFrame, caption, source=SOURCE_DEFAULT, formats=None, col_widths=None, font=9):
        """Caption above the table (thesis convention), source below."""
        self.tab_no += 1
        self._caption(f"Table {self.chapter}.{self.tab_no}: {caption}")
        formats = formats or {}
        t = self.d.add_table(rows=1, cols=len(df.columns))
        try:
            t.style = "Table Grid"
        except Exception:
            pass
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, c in enumerate(df.columns):
            cell = t.rows[0].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(c))
            run.bold = True
            run.font.size = Pt(font)
            _shade(cell, "D9E2F3")
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, c in enumerate(df.columns):
                cells[j].text = ""
                run = cells[j].paragraphs[0].add_run(fmt(row[c], formats.get(c)))
                run.font.size = Pt(font)
                if j > 0 and _is_num(row[c]):
                    cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if col_widths:
            for j, w in enumerate(col_widths):
                for r in t.rows:
                    r.cells[j].width = Cm(w)
        self._source(source)
        return f"Table {self.chapter}.{self.tab_no}"

    def page_break(self):
        self.d.add_page_break()

    def save(self, path):
        self.d.save(path)
        return path


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def fmt(v, f=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if f is not None:
        return f(v) if callable(f) else f.format(v)
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if _is_num(v):
        if float(v).is_integer():
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        return f"{v:,.3f}"
    return str(v)


def _shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _runs_with_bold(para, text, italic=False, bold=False, size=None, color=None):
    """Supports **bold** spans inside a paragraph string."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = para.add_run(part)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color
