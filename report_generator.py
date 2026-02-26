"""
ReportGenerator — Gera relatórios PDF profissionais com as métricas de detecção de marcas.

Usa FPDF2 para criar PDFs com:
- Cabeçalho com título e informações do vídeo
- Tabela de marcas com tempos e frequências
- Análise gerada pelo agente IA (opcional)
"""

from fpdf import FPDF
from datetime import datetime


class ReportGenerator:
    """Gera relatórios PDF com dados de detecção de marcas em vídeos de F1."""

    def __init__(self):
        self.pdf = None

    def _create_pdf(self):
        """Inicializa o PDF com configurações padrão."""
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.add_page()

    def _add_header(self, video_name: str):
        """Adiciona o cabeçalho do relatório."""
        # Title
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.set_text_color(200, 30, 30)
        self.pdf.cell(0, 15, "F1 Ad Report", new_x="LMARGIN", new_y="NEXT", align="C")

        # Subtitle
        self.pdf.set_font("Helvetica", "", 12)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 8, "Brand Visibility Analysis", new_x="LMARGIN", new_y="NEXT", align="C")

        # Line separator
        self.pdf.set_draw_color(200, 30, 30)
        self.pdf.set_line_width(0.8)
        self.pdf.line(20, self.pdf.get_y() + 2, 190, self.pdf.get_y() + 2)
        self.pdf.ln(8)

        # Video info
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(60, 60, 60)
        self.pdf.cell(0, 6, f"Video: {video_name}", new_x="LMARGIN", new_y="NEXT")
        self.pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(5)

    def _add_video_summary(self, metrics: dict):
        """Adiciona o resumo do vídeo."""
        info = metrics["video_info"]

        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(30, 30, 30)
        self.pdf.cell(0, 10, "Video Summary", new_x="LMARGIN", new_y="NEXT")

        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(60, 60, 60)

        summary_data = [
            ("Duration", f"{info['duration_seconds']}s ({info['duration_seconds']/60:.1f} min)"),
            ("FPS", str(info['fps'])),
            ("Total Frames", str(info['total_frames'])),
            ("Brands Detected", str(metrics['total_brands_detected'])),
            ("Most Frequent Brand", f"{metrics['most_frequent_brand']} ({metrics['most_frequent_count']} frames)"),
        ]

        for label, value in summary_data:
            self.pdf.set_font("Helvetica", "B", 10)
            self.pdf.cell(50, 7, f"{label}:", new_x="RIGHT")
            self.pdf.set_font("Helvetica", "", 10)
            self.pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

        self.pdf.ln(5)

    def _add_brand_table(self, metrics: dict):
        """Adiciona a tabela de marcas."""
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(30, 30, 30)
        self.pdf.cell(0, 10, "Brand Breakdown", new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(2)

        # Table header
        col_widths = [45, 30, 30, 30, 35]
        headers = ["Brand", "Time (s)", "Frames", "Detections", "Time %"]

        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(200, 30, 30)
        self.pdf.set_text_color(255, 255, 255)

        for i, header in enumerate(headers):
            self.pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        self.pdf.ln()

        # Sort brands by time_visible_seconds descending
        sorted_brands = sorted(
            metrics["brands"].items(),
            key=lambda x: x[1]["time_visible_seconds"],
            reverse=True
        )

        # Table rows
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(30, 30, 30)
        fill = False

        for brand_name, data in sorted_brands:
            if fill:
                self.pdf.set_fill_color(245, 245, 245)
            else:
                self.pdf.set_fill_color(255, 255, 255)

            row = [
                brand_name,
                str(data["time_visible_seconds"]),
                str(data["frames_visible"]),
                str(data["total_detections"]),
                f"{data['time_percentage']}%",
            ]

            for i, cell_val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.pdf.cell(col_widths[i], 7, cell_val, border=1, fill=True, align=align)
            self.pdf.ln()
            fill = not fill

        self.pdf.ln(5)

    def _add_highlight_brand(self, metrics: dict):
        """Adiciona destaque para a marca mais frequente."""
        brand = metrics["most_frequent_brand"]
        if not brand or brand not in metrics["brands"]:
            return

        data = metrics["brands"][brand]

        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(30, 30, 30)
        self.pdf.cell(0, 10, "Most Prominent Brand", new_x="LMARGIN", new_y="NEXT")

        # Highlight box
        self.pdf.set_fill_color(255, 240, 240)
        self.pdf.set_draw_color(200, 30, 30)
        y_start = self.pdf.get_y()
        self.pdf.rect(20, y_start, 170, 30, style="DF")

        self.pdf.set_xy(25, y_start + 3)
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.set_text_color(200, 30, 30)
        self.pdf.cell(0, 10, brand)

        self.pdf.set_xy(25, y_start + 14)
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(60, 60, 60)
        self.pdf.cell(
            0, 8,
            f"Visible for {data['time_visible_seconds']}s | "
            f"{data['time_percentage']}% of video | "
            f"{data['total_detections']} detections | "
            f"Avg confidence: {data['average_confidence']}"
        )

        self.pdf.set_y(y_start + 35)

    def _add_ai_analysis(self, ai_analysis: str):
        """Adiciona a análise gerada pelo agente IA."""
        if not ai_analysis:
            return

        self.pdf.add_page()
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.set_text_color(30, 30, 30)
        self.pdf.cell(0, 10, "AI Analysis (QWEN 4B)", new_x="LMARGIN", new_y="NEXT")

        # Thin red line
        self.pdf.set_draw_color(200, 30, 30)
        self.pdf.set_line_width(0.4)
        self.pdf.line(20, self.pdf.get_y(), 190, self.pdf.get_y())
        self.pdf.ln(5)

        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(40, 40, 40)

        # Process markdown-like text into PDF
        for line in ai_analysis.split("\n"):
            stripped = line.strip()

            if not stripped:
                self.pdf.ln(3)
                continue

            if stripped.startswith("# "):
                self.pdf.set_font("Helvetica", "B", 14)
                self.pdf.set_text_color(200, 30, 30)
                self.pdf.multi_cell(0, 7, stripped[2:])
                self.pdf.ln(2)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(40, 40, 40)

            elif stripped.startswith("## "):
                self.pdf.set_font("Helvetica", "B", 12)
                self.pdf.set_text_color(30, 30, 30)
                self.pdf.multi_cell(0, 7, stripped[3:])
                self.pdf.ln(2)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(40, 40, 40)

            elif stripped.startswith("### "):
                self.pdf.set_font("Helvetica", "BI", 11)
                self.pdf.set_text_color(60, 60, 60)
                self.pdf.multi_cell(0, 7, stripped[4:])
                self.pdf.ln(1)
                self.pdf.set_font("Helvetica", "", 10)
                self.pdf.set_text_color(40, 40, 40)

            elif stripped.startswith("**") and stripped.endswith("**"):
                self.pdf.set_font("Helvetica", "B", 10)
                self.pdf.multi_cell(0, 6, stripped.replace("**", ""))
                self.pdf.set_font("Helvetica", "", 10)

            elif stripped.startswith("- ") or stripped.startswith("* "):
                bullet_text = stripped[2:]
                # Remove inline bold markers for cleaner PDF output
                clean_text = bullet_text.replace("**", "")
                self.pdf.cell(8, 6, chr(8226))  # bullet character
                self.pdf.multi_cell(0, 6, f" {clean_text}")

            else:
                # Remove inline markdown bold for cleaner output
                clean_line = stripped.replace("**", "")
                self.pdf.multi_cell(0, 6, clean_line)

    def generate(self, metrics: dict, ai_analysis: str | None, video_name: str, output_path: str):
        """
        Gera o relatório PDF completo.
        
        Args:
            metrics: Dicionário de métricas do BrandTracker.
            ai_analysis: Texto da análise gerada pelo agente IA (ou None).
            video_name: Nome do arquivo de vídeo analisado.
            output_path: Caminho de saída para o PDF.
        """
        self._create_pdf()
        self._add_header(video_name)
        self._add_video_summary(metrics)
        self._add_brand_table(metrics)
        self._add_highlight_brand(metrics)
        self._add_ai_analysis(ai_analysis)

        # Footer
        self.pdf.set_y(-30)
        self.pdf.set_font("Helvetica", "I", 8)
        self.pdf.set_text_color(150, 150, 150)
        self.pdf.cell(0, 10, "Generated by F1 Ad Report - Brand Visibility Analysis System",
                      new_x="LMARGIN", new_y="NEXT", align="C")

        self.pdf.output(output_path)
