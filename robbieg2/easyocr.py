import easyocr

def find_all_text_with_bounding_boxes(path: str) -> [dict]:
    try:
        reader = easyocr.Reader(['en'])
        results = reader.readtext(path)
        processed_results = []
        for box in results:
            result = {
                "x": int(box[0][0][0]),
                "y": int(box[0][0][1]),
                "w": int(box[0][2][0] - box[0][0][0]),
                "h": int(box[0][2][1] - box[0][0][1]),
                "text": box[1],
                "confidence": float(box[2])
            }
            processed_results.append(result)
        return processed_results
    except Exception as e:
        print(f"EasyOCR failed: {str(e)}")
        return []
