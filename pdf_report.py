"""
pdf_report.py — Generates professional PDF reports with charts and brand metrics.

Uses FPDF2 for PDF layout and Matplotlib for data visualization:
- Page 1: Header, video summary cards, brand table, Top 5 ranking
- Page 2: Display time bar chart + screen time market share donut chart
- Page 3: Detection confidence chart (color-coded by confidence level)
- Page 4: AI-generated analysis (optional, requires LM Studio)
"""

import os
import tempfile
from fpdf import FPDF
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class ReportGenerator:
    """Generates PDF reports with charts from F1 brand detection data."""

    # Color palette
    RED = (200, 30, 30)
    DARK = (30, 30, 30)
    GRAY = (100, 100, 100)
    LIGHT_GRAY = (150, 150, 150)
    TEXT = (40, 40, 40)
    BG_LIGHT = (248, 248, 248)

    def __init__(self):
        self.pdf = None
        self._temp_files = []

    def _cleanup(self):
        """Remove temporary chart image files."""
        for f in self._temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        self._temp_files = []

    def _save_chart(self, fig) -> str:
        """Save a matplotlib figure to a temp file and return the path."""
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp.name, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        self._temp_files.append(tmp.name)
        return tmp.name

    def _create_pdf(self):
        """Inicializa o PDF."""
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.add_page()

    def _add_header(self, video_name: str):
        """Cabecalho do relatorio com design premium."""
        # Red accent bar at top
        self.pdf.set_fill_color(*self.RED)
        self.pdf.rect(0, 0, 210, 4, style="F")

        self.pdf.ln(10)

        # Title
        self.pdf.set_font("Helvetica", "B", 28)
        self.pdf.set_text_color(*self.RED)
        self.pdf.cell(0, 15, "F1 Ad Report", new_x="LMARGIN", new_y="NEXT", align="C")

        # Subtitle
        self.pdf.set_font("Helvetica", "", 13)
        self.pdf.set_text_color(*self.GRAY)
        self.pdf.cell(0, 8, "Brand Visibility Analysis", new_x="LMARGIN", new_y="NEXT", align="C")

        # Separator line
        self.pdf.set_draw_color(*self.RED)
        self.pdf.set_line_width(0.8)
        y = self.pdf.get_y() + 4
        self.pdf.line(30, y, 180, y)
        self.pdf.ln(10)

        # Video info box
        self.pdf.set_fill_color(*self.BG_LIGHT)
        self.pdf.set_draw_color(220, 220, 220)
        box_y = self.pdf.get_y()
        self.pdf.rect(20, box_y, 170, 18, style="DF")

        self.pdf.set_xy(25, box_y + 3)
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*self.DARK)
        self.pdf.cell(30, 6, "VIDEO:")
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(*self.TEXT)
        self.pdf.cell(80, 6, video_name)

        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*self.DARK)
        self.pdf.cell(20, 6, "DATE:")
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(*self.TEXT)
        self.pdf.cell(0, 6, datetime.now().strftime('%Y-%m-%d %H:%M'))

        self.pdf.set_y(box_y + 22)

    def _section_title(self, title: str):
        """Adiciona titulo de secao com estilo."""
        self.pdf.ln(3)
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(*self.RED)
        self.pdf.cell(5, 10, "", new_x="RIGHT")
        self.pdf.set_text_color(*self.DARK)
        self.pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

        # Thin accent line
        self.pdf.set_draw_color(*self.RED)
        self.pdf.set_line_width(0.5)
        self.pdf.line(20, self.pdf.get_y(), 60, self.pdf.get_y())
        self.pdf.ln(4)

    def _add_video_summary(self, metrics: dict):
        """Resumo do video com metricas-chave em cards."""
        info = metrics["video_info"]

        self._section_title("Video Summary")

        # Key metrics in a row of "cards"
        cards = [
            ("Duration", f"{info['duration_seconds']/60:.1f} min"),
            ("FPS", str(info['fps'])),
            ("Frames", f"{info['total_frames']:,}"),
            ("Brands", str(metrics['total_brands_detected'])),
        ]

        card_w = 40
        start_x = 15
        self.pdf.set_y(self.pdf.get_y() + 1)
        card_y = self.pdf.get_y()

        for i, (label, value) in enumerate(cards):
            x = start_x + i * (card_w + 4)

            # Card background
            self.pdf.set_fill_color(*self.BG_LIGHT)
            self.pdf.set_draw_color(220, 220, 220)
            self.pdf.rect(x, card_y, card_w, 20, style="DF")

            # Value
            self.pdf.set_xy(x, card_y + 2)
            self.pdf.set_font("Helvetica", "B", 16)
            self.pdf.set_text_color(*self.RED)
            self.pdf.cell(card_w, 10, value, align="C")

            # Label
            self.pdf.set_xy(x, card_y + 12)
            self.pdf.set_font("Helvetica", "", 8)
            self.pdf.set_text_color(*self.GRAY)
            self.pdf.cell(card_w, 6, label, align="C")

        self.pdf.set_y(card_y + 26)

    def _add_brand_table(self, metrics: dict):
        """Tabela de marcas com cores alternadas."""
        self._section_title("Brand Breakdown")

        col_widths = [45, 28, 28, 30, 25, 30]
        headers = ["Brand", "Time (s)", "Frames", "Detections", "Time %", "Confidence"]

        # Table header
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_fill_color(*self.RED)
        self.pdf.set_text_color(255, 255, 255)

        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 8, header, border=0, fill=True, align="C")
        self.pdf.ln()

        # Sort brands by time_visible_seconds descending
        sorted_brands = sorted(
            metrics["brands"].items(),
            key=lambda x: x[1]["time_visible_seconds"],
            reverse=True
        )

        self.pdf.set_font("Helvetica", "", 8)
        fill = False

        for brand_name, data in sorted_brands:
            if fill:
                self.pdf.set_fill_color(245, 245, 250)
            else:
                self.pdf.set_fill_color(255, 255, 255)

            self.pdf.set_text_color(*self.DARK)

            row = [
                brand_name,
                f"{data['time_visible_seconds']:.1f}",
                str(data["frames_visible"]),
                str(data["total_detections"]),
                f"{data['time_percentage']}%",
                f"{data['average_confidence']:.3f}",
            ]

            for i, cell_val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.pdf.cell(col_widths[i], 7, cell_val, border=0, fill=True, align=align)
            self.pdf.ln()

            # Subtle row separator
            self.pdf.set_draw_color(230, 230, 230)
            self.pdf.set_line_width(0.1)
            self.pdf.line(20, self.pdf.get_y(), 186, self.pdf.get_y())

            fill = not fill

        self.pdf.ln(3)

    def _add_highlight_brand(self, metrics: dict):
        """Top 5 Brand Ranking."""
        sorted_brands = sorted(
            metrics["brands"].items(),
            key=lambda x: x[1]["time_visible_seconds"],
            reverse=True
        )

        top5 = sorted_brands[:5]
        if not top5:
            return

        self._section_title("Top 5 Brand Ranking")

        # Ensure enough space — move to new page if needed
        if self.pdf.get_y() > 170:
            self.pdf.add_page()
            self.pdf.set_fill_color(*self.RED)
            self.pdf.rect(0, 0, 210, 4, style="F")
            self.pdf.ln(8)
            self._section_title("Top 5 Brand Ranking")

        self.pdf.set_auto_page_break(auto=False)

        # Medal colors for positions 1-5
        rank_colors = [
            (200, 30, 30),    # 1st — Red (champion)
            (180, 50, 50),    # 2nd — Dark red
            (140, 70, 70),    # 3rd — Medium red
            (120, 90, 90),    # 4th — Muted
            (100, 100, 100),  # 5th — Gray
        ]

        bg_colors = [
            (255, 230, 230),  # 1st
            (255, 238, 238),  # 2nd
            (255, 243, 243),  # 3rd
            (248, 245, 245),  # 4th
            (245, 245, 245),  # 5th
        ]

        box_height = 22
        spacing = 3

        for rank, (brand_name, data) in enumerate(top5):
            y = self.pdf.get_y()

            # Background box
            self.pdf.set_fill_color(*bg_colors[rank])
            self.pdf.set_draw_color(220, 220, 220)
            self.pdf.set_line_width(0.3)
            self.pdf.rect(20, y, 170, box_height, style="DF")

            # Colored accent bar on left
            self.pdf.set_fill_color(*rank_colors[rank])
            self.pdf.rect(20, y, 4, box_height, style="F")

            # Rank number
            self.pdf.set_xy(26, y + 2)
            self.pdf.set_font("Helvetica", "B", 18)
            self.pdf.set_text_color(*rank_colors[rank])
            self.pdf.cell(12, box_height - 4, f"#{rank + 1}", align="C")

            # Brand name
            self.pdf.set_xy(40, y + 2)
            self.pdf.set_font("Helvetica", "B", 13)
            self.pdf.set_text_color(*self.DARK)
            self.pdf.cell(80, 9, brand_name)

            # Stats line
            self.pdf.set_xy(40, y + 11)
            self.pdf.set_font("Helvetica", "", 8)
            self.pdf.set_text_color(*self.GRAY)
            self.pdf.cell(
                150, 7,
                f"{data['time_visible_seconds']}s  |  "
                f"{data['time_percentage']}% of video  |  "
                f"{data['total_detections']:,} detections  |  "
                f"Avg conf: {data['average_confidence']:.3f}"
            )

            # Time bar visualization (mini progress bar on the right)
            max_time = top5[0][1]["time_visible_seconds"]
            bar_pct = data["time_visible_seconds"] / max_time if max_time > 0 else 0
            bar_x = 140
            bar_w = 45
            bar_y = y + 4

            # Bar background
            self.pdf.set_fill_color(230, 230, 230)
            self.pdf.rect(bar_x, bar_y, bar_w, 5, style="F")

            # Bar fill
            self.pdf.set_fill_color(*rank_colors[rank])
            self.pdf.rect(bar_x, bar_y, bar_w * bar_pct, 5, style="F")

            # Percentage text
            self.pdf.set_xy(bar_x, bar_y + 5)
            self.pdf.set_font("Helvetica", "B", 7)
            self.pdf.set_text_color(*rank_colors[rank])
            self.pdf.cell(bar_w, 5, f"{data['time_percentage']}%", align="C")

            self.pdf.set_y(y + box_height + spacing)

        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.ln(3)

    def _generate_time_chart(self, metrics: dict) -> str:
        """Grafico de barras horizontais — tempo de exibicao por marca."""
        brands_data = metrics["brands"]
        sorted_brands = sorted(brands_data.items(), key=lambda x: x[1]["time_visible_seconds"])

        names = [b[0] for b in sorted_brands]
        times = [b[1]["time_visible_seconds"] for b in sorted_brands]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))

        # Color gradient from light to dark red
        colors = plt.cm.Reds(np.linspace(0.3, 0.85, len(names)))

        bars = ax.barh(names, times, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

        # Add value labels
        for bar, val in zip(bars, times):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}s', va='center', fontsize=8, color='#333333')

        ax.set_xlabel('Display Time (seconds)', fontsize=10, color='#555555')
        ax.set_title('Brand Visibility by Display Time', fontsize=13, fontweight='bold',
                      color='#222222', pad=15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.tick_params(colors='#555555', labelsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

        fig.tight_layout()
        return self._save_chart(fig)

    def _generate_pie_chart(self, metrics: dict) -> str:
        """Grafico de pizza — market share de tempo de tela."""
        brands_data = metrics["brands"]
        sorted_brands = sorted(brands_data.items(), key=lambda x: x[1]["time_visible_seconds"],
                                reverse=True)

        # Top brands + "Others"
        top_n = 6
        top_brands = sorted_brands[:top_n]
        others_time = sum(b[1]["time_visible_seconds"] for b in sorted_brands[top_n:])

        names = [b[0] for b in top_brands]
        times = [b[1]["time_visible_seconds"] for b in top_brands]

        if others_time > 0:
            names.append("Others")
            times.append(others_time)

        # Colors
        colors = ['#c81e1e', '#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#95a5a6']

        fig, ax = plt.subplots(figsize=(6, 5))

        wedges, texts, autotexts = ax.pie(
            times, labels=names, autopct='%1.1f%%',
            colors=colors[:len(names)],
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        )

        # Style text
        for text in texts:
            text.set_fontsize(9)
            text.set_color('#333333')
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Screen Time Market Share', fontsize=13, fontweight='bold',
                      color='#222222', pad=15)

        fig.tight_layout()
        return self._save_chart(fig)

    def _generate_confidence_chart(self, metrics: dict) -> str:
        """Grafico de barras — confianca media por marca."""
        brands_data = metrics["brands"]
        sorted_brands = sorted(brands_data.items(),
                                key=lambda x: x[1]["average_confidence"], reverse=True)

        names = [b[0] for b in sorted_brands]
        confs = [b[1]["average_confidence"] for b in sorted_brands]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))

        # Color based on confidence level
        colors = []
        for c in confs:
            if c >= 0.7:
                colors.append('#27ae60')  # green
            elif c >= 0.5:
                colors.append('#f39c12')  # orange
            else:
                colors.append('#e74c3c')  # red

        bars = ax.barh(names, confs, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

        # Add value labels
        for bar, val in zip(bars, confs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8, color='#333333')

        # Threshold lines
        ax.axvline(x=0.7, color='#27ae60', linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(x=0.5, color='#f39c12', linestyle='--', alpha=0.4, linewidth=1)

        ax.set_xlabel('Average Confidence', fontsize=10, color='#555555')
        ax.set_title('Detection Confidence by Brand', fontsize=13, fontweight='bold',
                      color='#222222', pad=15)
        ax.set_xlim(0, 1.0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
        ax.tick_params(colors='#555555', labelsize=9)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', label='High (>0.7)'),
            Patch(facecolor='#f39c12', label='Medium (0.5-0.7)'),
            Patch(facecolor='#e74c3c', label='Low (<0.5)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.8)

        fig.tight_layout()
        return self._save_chart(fig)

    def _add_charts(self, metrics: dict):
        """Adiciona pagina com graficos."""
        # --- Page 2: Bar chart + Pie chart ---
        self.pdf.add_page()

        # Red accent bar
        self.pdf.set_fill_color(*self.RED)
        self.pdf.rect(0, 0, 210, 4, style="F")
        self.pdf.ln(8)

        self._section_title("Visual Analytics")

        # Time bar chart
        time_chart = self._generate_time_chart(metrics)
        self.pdf.image(time_chart, x=15, w=180)
        self.pdf.ln(8)

        # Pie chart
        pie_chart = self._generate_pie_chart(metrics)
        chart_w = 110
        x_center = (210 - chart_w) / 2
        self.pdf.image(pie_chart, x=x_center, w=chart_w)

    def _add_ai_analysis(self, ai_analysis: str):
        """Analise gerada pelo agente IA."""
        if not ai_analysis:
            return

        self.pdf.add_page()

        # Red accent bar
        self.pdf.set_fill_color(*self.RED)
        self.pdf.rect(0, 0, 210, 4, style="F")
        self.pdf.ln(8)

        self._section_title("AI Analysis (QWEN)")

        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(*self.TEXT)

        for line in ai_analysis.split("\n"):
            stripped = line.strip()

            if not stripped:
                self.pdf.ln(3)
                continue

            if stripped.startswith("# "):
                self.pdf.set_font("Helvetica", "B", 14)
                self.pdf.set_text_color(*self.RED)
                self.pdf.multi_cell(0, 7, stripped[2:])
                self.pdf.ln(2)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(*self.TEXT)

            elif stripped.startswith("## "):
                self.pdf.set_font("Helvetica", "B", 12)
                self.pdf.set_text_color(*self.DARK)
                self.pdf.multi_cell(0, 7, stripped[3:])
                self.pdf.ln(2)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(*self.TEXT)

            elif stripped.startswith("### "):
                self.pdf.set_font("Helvetica", "BI", 11)
                self.pdf.set_text_color(80, 80, 80)
                self.pdf.multi_cell(0, 7, stripped[4:])
                self.pdf.ln(1)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(*self.TEXT)

            elif stripped.startswith("**") and stripped.endswith("**"):
                self.pdf.set_font("Helvetica", "B", 10)
                self.pdf.multi_cell(0, 6, stripped.replace("**", ""))
                self.pdf.set_font("Helvetica", "", 10)

            elif stripped.startswith("- ") or stripped.startswith("* "):
                bullet_text = stripped[2:].replace("**", "")
                self.pdf.cell(8, 6, chr(8226))
                self.pdf.multi_cell(0, 6, f" {bullet_text}")

            else:
                clean_line = stripped.replace("**", "")
                self.pdf.multi_cell(0, 6, clean_line)

    def generate(self, metrics: dict, ai_analysis: str | None, video_name: str, output_path: str):
        """
        Gera o relatorio PDF completo com graficos.

        Args:
            metrics: Dicionario de metricas do BrandTracker.
            ai_analysis: Texto da analise gerada pelo agente IA (ou None).
            video_name: Nome do arquivo de video analisado.
            output_path: Caminho de saida para o PDF.
        """
        try:
            self._create_pdf()

            # Page 1: Header, Summary, Table, Highlight
            self._add_header(video_name)
            self._add_video_summary(metrics)
            self._add_brand_table(metrics)
            self._add_highlight_brand(metrics)

            # Pages 2-3: Charts
            self._add_charts(metrics)

            # Page 4: AI Analysis (optional)
            self._add_ai_analysis(ai_analysis)

            self.pdf.output(output_path)
        finally:
            self._cleanup()
