import os
import sys
from pathlib import Path
import pymupdf as fitz
from rapidocr_onnxruntime import RapidOCR


def extract_pdf(pdf_path_str: str, output_folder: str = "output_extracted"):
    pdf_path = Path(pdf_path_str.strip('"').strip("'"))
    if not pdf_path.exists():
        print(f"[!] Error: File not found at '{pdf_path}'")
        return

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[*] Extracting PDF: {pdf_path.name}")
    print(f"[*] Output Directory: {out_dir.resolve()}")
    print("=" * 60 + "\n")

    doc = fitz.open(str(pdf_path))
    ocr = RapidOCR()

    extracted_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        p_num = page_num + 1
        print(f"--- Processing Page {p_num}/{len(doc)} ---")

        # 1. Extract Digital Native Text
        native_text = (page.get_text() or "").strip()

        # 2. Render and save high-resolution Page Image (captures vector diagrams & charts)
        page_img_path = images_dir / f"page_{p_num}.png"
        pix = page.get_pixmap(dpi=150)
        pix.save(str(page_img_path))

        # 3. Extract any raw embedded bitmap images on page
        embedded_imgs = []
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list, start=1):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            raw_img_path = images_dir / f"page_{p_num}_raw_img_{img_idx}.{image_ext}"
            with open(raw_img_path, "wb") as f:
                f.write(image_bytes)
            embedded_imgs.append(raw_img_path.name)

        # 4. Run RapidOCR on the page image to catch text inside diagrams/charts
        ocr_result, _ = ocr(str(page_img_path))
        ocr_lines = [item[1].strip() for item in ocr_result if item[1].strip()] if ocr_result else []
        ocr_text = "\n".join(ocr_lines)

        print(f"  [+] Page {p_num} image saved: {page_img_path.name}")
        if embedded_imgs:
            print(f"  [+] Extracted {len(embedded_imgs)} embedded image(s): {', '.join(embedded_imgs)}")
        print(f"  [+] Text: {len(native_text)} chars | OCR Text: {len(ocr_text)} chars")

        extracted_pages.append({
            "page_num": p_num,
            "page_image": f"images/{page_img_path.name}",
            "embedded_images": [f"images/{img}" for img in embedded_imgs],
            "native_text": native_text,
            "ocr_text": ocr_text
        })

    doc.close()

    # Save full text dump to output_extracted/extracted_text.txt
    text_dump_path = out_dir / "extracted_text.txt"
    with open(text_dump_path, "w", encoding="utf-8") as f:
        for p in extracted_pages:
            f.write(f"\n{'='*50}\nPAGE {p['page_num']}\n{'='*50}\n\n")
            f.write(f"--- NATIVE TEXT ---\n{p['native_text']}\n\n")
            f.write(f"--- OCR TEXT (FROM DIAGRAMS/IMAGES) ---\n{p['ocr_text']}\n\n")

    # Generate an interactive HTML Viewer
    html_preview_path = out_dir / "preview.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Extraction Results: {pdf_path.name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #090c15; color: #f8fafc; margin: 0; padding: 2rem; }}
        h1 {{ color: #ffe814; font-size: 1.8rem; border-bottom: 2px solid #334155; padding-bottom: 1rem; }}
        .summary {{ background: #1e293b; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #ff2d87; }}
        .page-card {{ background: #111827; border: 2px solid #334155; border-radius: 12px; margin-bottom: 2rem; overflow: hidden; display: grid; grid-template-columns: 450px 1fr; gap: 1.5rem; padding: 1.5rem; }}
        .img-col {{ display: flex; flex-direction: column; gap: 1rem; }}
        .img-col img {{ width: 100%; border-radius: 8px; border: 2px solid #000; background: #fff; }}
        .text-col {{ display: flex; flex-direction: column; gap: 1rem; overflow-y: auto; max-height: 700px; }}
        .text-box {{ background: #090c15; border: 1px solid #334155; border-radius: 6px; padding: 1rem; font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; line-height: 1.5; }}
        .badge {{ display: inline-block; background: #ff2d87; color: #fff; font-weight: bold; font-size: 0.75rem; padding: 0.2rem 0.6rem; border-radius: 4px; margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
    <h1>Extracted Content: {pdf_path.name}</h1>
    <div class="summary">
        <div><strong>Total Pages Processed:</strong> {len(extracted_pages)}</div>
        <div><strong>Images Directory:</strong> <code>{images_dir.resolve()}</code></div>
        <div><strong>Raw Text File:</strong> <code>{text_dump_path.resolve()}</code></div>
    </div>
"""

    for p in extracted_pages:
        html_content += f"""
    <div class="page-card">
        <div class="img-col">
            <div><span class="badge">PAGE {p['page_num']} DIAGRAM / IMAGE</span></div>
            <a href="{p['page_image']}" target="_blank">
                <img src="{p['page_image']}" alt="Page {p['page_num']}">
            </a>
        </div>
        <div class="text-col">
            <div>
                <span class="badge" style="background:#2be26c; color:#000;">NATIVE EXTRACTED TEXT</span>
                <div class="text-box">{p['native_text'] if p['native_text'] else '<i>(No native digital text on this page)</i>'}</div>
            </div>
            <div>
                <span class="badge" style="background:#ffe814; color:#000;">DIAGRAM & OCR TEXT</span>
                <div class="text-box">{p['ocr_text'] if p['ocr_text'] else '<i>(No text detected by OCR)</i>'}</div>
            </div>
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    with open(html_preview_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 60)
    print(f"[OK] EXTRACTION COMPLETE!")
    print(f"[*] All page images saved to: {images_dir.resolve()}")
    print(f"[*] Full text dump saved to:  {text_dump_path.resolve()}")
    print(f"[*] HTML Viewer created at:   {html_preview_path.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Prompt user if no argument is passed
        target = input("Enter the path to your PDF file: ").strip()

    if target:
        extract_pdf(target)
    else:
        print("No file specified.")
