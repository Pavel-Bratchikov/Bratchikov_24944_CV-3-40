import cv2
import pytesseract
import numpy as np
import re
import sys

def warp_document(image_path, debug=False):
    """Пытается найти контур документа и сделать перспективную трансформацию.
    Если контур не найден — возвращает оригинальное изображение с предупреждением.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Файл не найден или не удалось открыть: {image_path}")

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # небольшое размытие, чтобы сгладить шум
    gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

    # улучшаем контрасты/границы:
    # используем Canny с адаптивными порогами
    v = np.median(gray_blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray_blur, lower, upper)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]  # топ-20

    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        # Попытка 2: детекция больших прямоугольников на бинаризованном изображении
        _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:10]
        for c in contours2:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

    if doc_cnt is None:
        if debug:
            print("Внимание: не найден 4-угольный контур документа — возвращаю оригинал.")
        return orig  # fallback: оригинал

    pts = doc_cnt.reshape(4, 2).astype("float32")
    # сортировка точек (tl, tr, br, bl)
    s = pts.sum(axis=1)
    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # защита от нулевого размера
    if maxWidth < 10 or maxHeight < 10:
        return orig

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    return warped

def _preprocess_for_ocr(img_gray):
    """Набор вариантов предобработки: масштабирование, Otsu, морфология."""
    # увеличим в 2 раза по каждой оси — MRZ часто маленькая
    scale = 2.0
    img = cv2.resize(img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # сначала Otsu
    _, th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # инверсия если фон тёмный (MRZ — чёрный текст на светлом фоне)
    # подсчёт яркости: если среднее значение малое — инвертируем
    if np.mean(th_otsu) < 127:
        th_otsu = cv2.bitwise_not(th_otsu)

    # небольшое морфологическое открытие, чтобы убрать шумные точки
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(th_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def _try_ocr_variants(mrz_img):
    """Пробуем несколько конфигураций Tesseract (разные PSM), возвращаем текст."""
    # Список psm в порядке пробования
    psms = ["6", "4", "11", "3"]  # 6 = uniform block of text, 4 = single column, 11 = sparse text
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    results = []
    for psm in psms:
        config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        try:
            txt = pytesseract.image_to_string(mrz_img, config=config)
            if txt and any(ch.isalnum() or ch == "<" for ch in txt):
                results.append((psm, txt))
                # если получили две строки похожие на MRZ — можно прекратить
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                candidate = [l for l in lines if re.fullmatch(r'[A-Z0-9<]{10,}', l)]
                if len(candidate) >= 2:
                    return txt  # явно успешный результат
        except Exception as e:
            # не фатально — пробуем дальше
            continue
    # если явного успеха нет — возвращаем наиболее "длинный" найденный результат
    if results:
        best = max(results, key=lambda r: len(r[1]))
        return best[1]
    return ""

def extract_mrz_text(image, show_preview=False, debug=False):
    """
    Надёжное извлечение MRZ:
    - Берёт нижнюю часть выпрямлённого изображения (с запасом).
    - Применяет предобработку (resize, Otsu, морфология).
    - Пытается OCR с разными PSM.
    - Возвращает список строк MRZ (2 строки) или пустой список.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    # MRZ обычно занимает нижние ~20-30% документа. Берём с запасом.
    start_ratio = 0.60  # можно уменьшить, если MRZ сверху
    mrz_zone = gray[int(h * start_ratio):, :]

    # если зона слишком мелкая по высоте, расширим вверх
    if mrz_zone.shape[0] < 40 and h > 80:
        mrz_zone = gray[int(h * max(0, start_ratio - 0.15)): , :]

    preproc = _preprocess_for_ocr(mrz_zone)

    if show_preview:
        cv2.imshow("MRZ preprocessed", preproc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # пробуем OCR на предобработанном изображении
    text = _try_ocr_variants(preproc)

    # очистка: заменяем похожие символы и убираем пробелы внутри строк
    if text:
        # заменяем строчные на верхние и исправляем похожие символы
        text = text.upper()
        text = text.replace(" ", "")
        # частые ошибки: 0 <-> O, 1 <-> I, 5 <-> S
        text = text.replace("O", "O")  # тут ничего не меняем, но можно добавить правила по необходимости

    # Разделяем на строки и фильтруем
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Оставляем только строки из разрешённых символов
    allowed_re = re.compile(r'^[A-Z0-9<]+$')
    clean_lines = [re.sub(r'[^A-Z0-9<]', '', l) for l in lines]
    clean_lines = [l for l in clean_lines if allowed_re.match(l) and len(l) >= 30]

    # Если не нашли две строки длинных, попробуем более мягкие критерии:
    if len(clean_lines) < 2:
        # допускаем строки >= 25
        candidates = [l for l in [re.sub(r'[^A-Z0-9<]', '', s) for s in lines] if len(l) >= 25]
        if len(candidates) >= 2:
            clean_lines = candidates[:2]

    # Если всё ещё одна длинная строка (например OCR слил строки), попробуем разделить
    if len(clean_lines) == 1:
        single = clean_lines[0]
        if 60 <= len(single) <= 100:
            # попробуем разделить пополам (простейший эвристический метод)
            mid = len(single) // 2
            # найти ближайший символ '<' к середине для корректного раздела
            left = single[:mid]
            right = single[mid:]
            # если в правой части в начале есть много '<', сдвинем границу
            # в общем — ищем границу, чтобы обе части >=25 символов
            for shift in range(-10, 11):
                i = mid + shift
                if i <= 20 or i >= len(single) - 20:
                    continue
                a, b = single[:i], single[i:]
                if len(a) >= 25 and len(b) >= 25:
                    clean_lines = [a, b]
                    break

    # если всё ещё пусто, возвращаем пустой список
    return clean_lines

def parse_mrz(mrz_input):
    """Разбирает MRZ. Принимает либо список из 2 строк, либо строку (объединённую)."""
    if isinstance(mrz_input, str):
        # попытаемся получить список
        lines = [line.strip() for line in mrz_input.splitlines() if line.strip()]
    else:
        lines = mrz_input

    if len(lines) < 2:
        raise ValueError("Не удалось выделить две строки MRZ для разбора")

    line1, line2 = lines[0], lines[1]

    # Для защиты от коротких/ошибочных строк — расширяем до ожидаемой длины, если надо
    # Стандарт для паспортного MRZ: 44 символа в каждой строке.
    def pad_or_trim(s, length=44):
        s = re.sub(r'[^A-Z0-9<]', '', s.upper())
        if len(s) > length:
            return s[:length]
        return s.ljust(length, '<')

    line1p = pad_or_trim(line1, 44)
    line2p = pad_or_trim(line2, 44)

    doc_type = line1p[0:2]
    country = line1p[2:5]
    name_raw = line1p[5:44]
    name_parts = name_raw.split("<<")
    last_name = name_parts[0].replace("<", " ").strip()
    first_name = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""

    passport_number = line2p[0:9].replace("<", "").strip()
    passport_number_check = line2p[9]  # контрольная цифра (можно вычислить и проверить)
    nationality = line2p[10:13]
    birth_date = line2p[13:19]
    birth_date_check = line2p[19]
    sex = line2p[20]
    expiry_date = line2p[21:27]
    expiry_date_check = line2p[27]

    # Доп. поля: личный номер (optional)
    personal_number = line2p[28:42].replace("<", "").strip()
    personal_number_check = line2p[42]

    return {
        "document_type": doc_type,
        "country": country,
        "last_name": last_name,
        "first_name": first_name,
        "passport_number": passport_number,
        "passport_number_check": passport_number_check,
        "nationality": nationality,
        "birth_date": birth_date,
        "birth_date_check": birth_date_check,
        "sex": sex,
        "expiry_date": expiry_date,
        "expiry_date_check": expiry_date_check,
        "personal_number": personal_number,
        "personal_number_check": personal_number_check
    }

def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input("Введите имя файла с фото документа: ").strip()

    # Просим показать превью — иногда удобно отключить
    preview_answer = input("Показывать промежуточные окна предпросмотра? (y/N): ").strip().lower()
    show_preview = preview_answer == 'y'

    try:
        print("[1/4] Выпрямление документа...")
        warped = warp_document(image_path, debug=True)

        print("[2/4] Извлечение MRZ...")
        mrz_lines = extract_mrz_text(warped, show_preview=show_preview, debug=True)
        if not mrz_lines:
            print("Не удалось извлечь MRZ — попробуйте лучше сфотографировать документ (ровнее, без бликов).")
            return

        print("Найденные строки MRZ:")
        for i, l in enumerate(mrz_lines):
            print(f"{i+1}: {l}")

        if len(mrz_lines) < 2:
            print("Обнаружена только одна строка MRZ — попробуйте перекомпозировать фото или увеличить зону захвата.")
            return

        print("[3/4] Разбор MRZ...")
        info = parse_mrz(mrz_lines)

        print("\n[4/4] Результат:")
        for k, v in info.items():
            print(f"{k}: {v}")

    except Exception as e:
        print("Ошибка:", str(e))

if __name__ == "__main__":
    main()