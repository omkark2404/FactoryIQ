import re

class OCRTextCleaner:
    """Utilities for cleaning raw OCR text output."""
    
    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Normalizes OCR text:
        - Removes redundant control characters
        - Fixes broken line breaks and multi-space gaps
        - Standardizes temperature degree symbols
        """
        if not raw_text:
            return ""
            
        # Replace non-breaking spaces and special quotes
        text = raw_text.replace('\xa0', ' ').replace('’', "'").replace('”', '"').replace('“', '"')
        
        # Standardize degree symbol
        text = re.sub(r'(\d+)\s*deg(rees)?\s*c', r'\1°C', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*C\b', r'\1°C', text)
        text = text.replace('°°', '°')

        # Replace multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
