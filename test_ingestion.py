from pathlib import Path
from PIL import Image, ImageDraw
from core.ingestion.factory import DocumentParserFactory

def run_test():
    print("🚀 Initializing DocumentParserFactory...")
    factory = DocumentParserFactory()

    # 1. Test Text Parser
    sample_txt = Path("sample_test.txt")
    sample_txt.write_text("Hello! This is a test document for our Autonomous RAG Factory.\nAnti-hallucination is enabled.", encoding="utf-8")
    
    doc_txt = factory.parse_file(sample_txt)
    print(f"\n[TXT Test] Parsed: {doc_txt.filename}")
    print(f"Content:\n{doc_txt.text_content}")
    print(f"Metadata: {doc_txt.metadata}")

    # 2. Test Image / OCR Parser (We create a quick image with text)
    sample_img = Path("sample_image.png")
    img = Image.new('RGB', (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((15, 40), "SECRET CODE: 4829 - RAG OCR WORKING", fill=(0, 0, 0))
    img.save(sample_img)

    doc_img = factory.parse_file(sample_img)
    print(f"\n[OCR Image Test] Parsed: {doc_img.filename}")
    print(f"OCR Extracted Content:\n{doc_img.text_content}")
    print(f"Metadata: {doc_img.metadata}")

    # Clean up temporary test files
    sample_txt.unlink(missing_ok=True)
    sample_img.unlink(missing_ok=True)
    print("\n✅ All ingestion tests passed successfully!")

if __name__ == "__main__":
    run_test()
