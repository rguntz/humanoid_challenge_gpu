from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

COLS, ROWS = 10, 7        # number of squares -> 9x6 inner corners
SQUARE = 20 * mm          # square size

W, H = landscape(A4)
out = Path(__file__).parent / "checkerboard_10x7_20mm_A4.pdf"
c = canvas.Canvas(str(out), pagesize=(W, H))

# Centre the board on the page
x0 = (W - COLS * SQUARE) / 2
y0 = (H - ROWS * SQUARE) / 2

for r in range(ROWS):
    for k in range(COLS):
        if (r + k) % 2 == 0:
            c.rect(x0 + k * SQUARE, y0 + r * SQUARE, SQUARE, SQUARE,
                   stroke=0, fill=1)

c.setFont("Helvetica", 8)
c.drawString(15 * mm, 12 * mm,
             f"Checkerboard {COLS}x{ROWS} squares ({COLS-1}x{ROWS-1} inner corners), "
             f"{SQUARE/mm:.0f} mm squares. Print at 100% / actual size.")
c.save()