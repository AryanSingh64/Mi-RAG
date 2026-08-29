import os
from PIL import Image, ImageDraw

def create_lightning_icon(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size / 24.0
    pts = [
        (13 * s, 2 * s),
        (3 * s, 14 * s),
        (12 * s, 14 * s),
        (11 * s, 22 * s),
        (21 * s, 10 * s),
        (12 * s, 10 * s),
        (13 * s, 2 * s)
    ]
    draw.polygon(pts, fill=(255, 45, 135, 255), outline=(0, 0, 0, 255), width=max(1, int(1.5 * s)))
    return img

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="64" height="64">
  <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill="#ff2d87" stroke="#000000" stroke-width="1.5" stroke-linejoin="round"/>
</svg>"""

for d in ['a:/RAG/public/static/assets', 'a:/RAG/web/static/assets', 'a:/RAG/public']:
    os.makedirs(d, exist_ok=True)
    with open(f'{d}/favicon.svg', 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    icon32 = create_lightning_icon(32)
    icon32.save(f'{d}/favicon.png')
    
    icon180 = create_lightning_icon(180)
    icon180.save(f'{d}/apple-touch-icon.png')
    
    icon64 = create_lightning_icon(64)
    icon64.save(f'{d}/favicon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64)])

print("Pink lightning favicons generated successfully across all asset directories!")
