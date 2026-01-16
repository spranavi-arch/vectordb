
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import logging

logger = logging.getLogger(__name__)


class OCRService:
    
    def extract_text(self, file_bytes: bytes, content_type: str) -> list[tuple[int, str]]:

        results = []

        if content_type == "application/pdf":
            # Convert PDF to list of PIL Image objects
            images = convert_from_bytes(file_bytes)
            
            # Extract text from each page
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img)
                results.append((i + 1, text))  # 1-indexed page numbers

        else:
            # Treat as image file
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            results.append((1, text))  # Single page

        logger.info(f"OCR extracted text from {len(results)} pages")
        return results
