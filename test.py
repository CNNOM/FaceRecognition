import os
import json
from deepface import DeepFace


def face_verify(img_1, img_2):
    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2, model_name='Facenet512')

        with open('result.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        return result_dict
    except Exception as _ex:
        return _ex


def compare_with_folder(img_1, folder_path):
    results = []
    for filename in os.listdir(folder_path):
        img_2 = os.path.join(folder_path, filename)
        if os.path.isfile(img_2):  # Проверка, что это файл, а не подпапка
            result = face_verify(img_1, img_2)
            results.append({
                'image': filename,
                'result': result
            })

    return results


# Пример использования
img_1 = 'faceBPItest/img_3.png'
folder_path = 'faceBPI'
comparison_results = compare_with_folder(img_1, folder_path)

# Вывод результатов
for result in comparison_results:
    print(f"Image: {result['image']}, Result: {result['result']}")