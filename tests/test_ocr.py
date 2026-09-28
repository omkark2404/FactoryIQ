import pytest
from models.ocr.clean import OCRTextCleaner
from models.ocr.parser import OCRDocumentParser

def test_ocr_cleaner():
    raw = "Batch: RB-2041   Machine: Press-04   Operating Temp: 184 deg C\n\n\nRejected: 126"
    clean = OCRTextCleaner.clean_text(raw)
    assert "184°C" in clean
    assert "\n\n\n" not in clean

def test_ocr_parser():
    sample_text = "Inspection Report Batch: RB-2041 Machine: Press-04 Operating Temp: 184°C Shift: B Rejected Units: 126 Defect: Surface crack"
    parsed = OCRDocumentParser.parse_metadata(sample_text)
    assert parsed["batch_id"] == "RB-2041"
    assert parsed["machine_id"] == "Press-04"
    assert parsed["operating_temp_c"] == 184.0
    assert parsed["shift"] == "B"
    assert parsed["rejected_units"] == 126
