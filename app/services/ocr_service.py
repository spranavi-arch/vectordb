
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import logging

logger = logging.getLogger(__name__)


class OCRService:

    def is_pdf(self, file_bytes: bytes, content_type: str) -> bool:
        """
        Determine whether the uploaded file is a PDF.

        Client-supplied content_type headers are unreliable (browsers/clients
        often send generic types like application/octet-stream), so sniff the
        actual file signature first and only fall back to the declared
        content_type if sniffing is inconclusive (e.g. empty file).
        """
        if file_bytes[:5] == b"%PDF-":
            return True
        if file_bytes[:4] == b"\x89PNG" or file_bytes[:3] == b"\xff\xd8\xff" or file_bytes[:6] in (b"GIF87a", b"GIF89a"):
            return False
        return content_type == "application/pdf"

    def extract_text(self, file_bytes: bytes, content_type: str) -> list[tuple[int, str]]:

        results = []

        if self.is_pdf(file_bytes, content_type):
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
