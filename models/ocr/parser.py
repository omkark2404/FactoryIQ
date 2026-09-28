import re

class OCRDocumentParser:
    """Parses key manufacturing fields (Batch ID, Machine ID, Temp, Shift, Defect Count) from OCR text."""
    
    @staticmethod
    def parse_metadata(text: str) -> dict:
        """Extracts structured key-value manufacturing metrics from clean OCR text."""
        metadata = {
            "batch_id": None,
            "machine_id": None,
            "operating_temp_c": None,
            "shift": None,
            "rejected_units": None,
            "defect_type": None
        }
        
        # Batch ID matching (e.g. RB-2041, RB2041, BATCH: RB-2041)
        batch_match = re.search(r'\b(RB-?\d{4})\b', text, re.IGNORECASE)
        if batch_match:
            raw_b = batch_match.group(1).upper()
            metadata["batch_id"] = raw_b if "-" in raw_b else f"RB-{raw_b[2:]}"
            
        # Machine ID matching (e.g. Press-04, PRESS 04)
        machine_match = re.search(r'\b(Press-?\d{2})\b', text, re.IGNORECASE)
        if machine_match:
            raw_m = machine_match.group(1).title()
            metadata["machine_id"] = raw_m if "-" in raw_m else f"Press-{raw_m[5:]}"

        # Operating Temperature matching (e.g. 184°C, 184 C, 184.5°C)
        temp_match = re.search(r'(\d{2,3}(?:\.\d)?)\s*°?C\b', text, re.IGNORECASE)
        if temp_match:
            metadata["operating_temp_c"] = float(temp_match.group(1))

        # Shift matching (Shift A, Shift B, Shift C, Shift: B)
        shift_match = re.search(r'Shift\s*[:\-]?\s*([A-C])\b', text, re.IGNORECASE)
        if shift_match:
            metadata["shift"] = shift_match.group(1).upper()

        # Rejected Units matching (e.g. Rejected Units: 126, 126 units)
        rejected_match = re.search(r'(?:Rejected\s*Units?|Rejections?)\s*[:\-]?\s*(\d+)', text, re.IGNORECASE)
        if rejected_match:
            metadata["rejected_units"] = int(rejected_match.group(1))

        # Defect Type matching
        defect_match = re.search(r'Defect(?:\s*Type)?\s*[:\-]?\s*([A-Za-z\s]+)(?:\n|$|\.)', text, re.IGNORECASE)
        if defect_match:
            metadata["defect_type"] = defect_match.group(1).strip()

        return metadata
