# MRZ OCR Extractor

This Python project extracts and parses the **Machine Readable Zone (MRZ)** from passport or ID document images. It performs document warping, preprocessing, and OCR using **OpenCV** and **Tesseract** to reliably extract the MRZ text and parse it into structured fields.

---

## Features

* Automatically detects and warps the document to a top-down perspective.
* Extracts the MRZ region from the lower part of the document.
* Applies preprocessing for OCR (resizing, thresholding, morphological operations).
* Supports multiple Tesseract PSM (Page Segmentation Modes) for better OCR accuracy.
* Parses MRZ into structured fields:

  * Document type
  * Country
  * Last name / First name
  * Passport number
  * Nationality
  * Date of birth
  * Sex
  * Expiration date
  * Personal number (optional)

---

## Requirements

* Python 3.8+
* OpenCV
* NumPy
* pytesseract
* Tesseract OCR engine installed on your system

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/mrz-ocr-extractor.git
cd mrz-ocr-extractor
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install Python dependencies:**

```bash
pip install opencv-python numpy pytesseract
```

4. **Install Tesseract OCR engine:**

* **Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

* **macOS (using Homebrew):**

```bash
brew install tesseract
```

* **Windows:**

  * Download the installer from [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract) or [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
  * Add the installation path to your system `PATH`.

5. **Verify Tesseract installation:**

```bash
tesseract --version
```

---

## Usage

Run the script from the command line:

```bash
python mrz_extractor.py path/to/document_image.jpg
```

* You will be prompted whether to show preview windows (preprocessed MRZ). Type `y` for yes or `Enter` for no.
* The program will output the detected MRZ lines and the parsed structured information.

### Example Output

```
[1/4] Warping document...
[2/4] Extracting MRZ...
Detected MRZ lines:
1: P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<
2: L898902C36UTO7408122F1204159ZE184226B<<<<<10

[3/4] Parsing MRZ...

[4/4] Result:
document_type: P<
country: UTO
last_name: ERIKSSON
first_name: ANNA MARIA
passport_number: L898902C3
passport_number_check: 6
nationality: UTO
birth_date: 740812
birth_date_check: 2
sex: F
expiry_date: 120415
expiry_date_check: 9
personal_number: ZE184226B
personal_number_check: 1
```

---

## Notes

* Ensure the document photo is clear, well-lit, and relatively flat.
* The MRZ region is usually located at the bottom 20–30% of the passport or ID.
* If MRZ extraction fails, try repositioning the photo, increasing capture area, or reducing glare.

