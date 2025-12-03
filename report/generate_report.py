# report/generate_report.py
from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, "Alzheimer's Diagnosis Report", ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_report(pred, prob, gradcam_path, shap_path, mmse, age, patient_name="Anonymous", treatment=""):
    pdf = PDF()

    # Add fonts
    font_dir = os.path.dirname(__file__)
    regular = os.path.join(font_dir, "DejaVuSans.ttf")
    bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
    if not os.path.exists(regular) or not os.path.exists(bold):
        raise FileNotFoundError("DejaVu fonts missing in report/")

    pdf.add_font('DejaVu', '', regular, uni=True)
    pdf.add_font('DejaVu', 'B', bold, uni=True)

    pdf.add_page()
    pdf.set_font('DejaVu', '', 11)

    # Patient Info
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Patient Information", ln=1)
    pdf.set_font('DejaVu', '', 11)
    pdf.cell(0, 8, f"Name: {patient_name}", ln=1)
    pdf.cell(0, 8, f"Age: {age} years", ln=1)
    pdf.cell(0, 8, f"MMSE Score: {mmse}/30", ln=1)
    pdf.ln(8)

    # Diagnosis
    pdf.set_font('DejaVu', 'B', 14)
    color = {'CN': (0,150,0), 'MCI': (255,165,0), 'AD': (200,0,0)}
    r,g,b = color.get(pred, (0,0,0))
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 12, f"Diagnosis: {pred} ({prob:.1f}% confidence)", ln=1, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Grad-CAM
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Brain Activation (Grad-CAM)", ln=1)
    if os.path.exists(gradcam_path):
        pdf.image(gradcam_path, x=25, w=160)
    pdf.ln(15)

    # SHAP
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Risk Factors (SHAP)", ln=1)
    if os.path.exists(shap_path):
        pdf.image(shap_path, x=25, w=160)
    pdf.ln(15)

    # Treatment Plan
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 10, "Treatment Plan", ln=1)
    pdf.set_font('DejaVu', '', 11)
    pdf.multi_cell(0, 7, treatment or "No treatment specified.")
    pdf.ln(10)

    # Save
    os.makedirs("results", exist_ok=True)
    path = "results/alzheimers_report.pdf"
    pdf.output(path)
    return path


