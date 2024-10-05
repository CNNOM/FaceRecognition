from deepface import DeepFace
import json

def face_verify(img_1, img_2):

    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2, model_name='Facenet512')

        with open('result.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        return result_dict
    except Exception as _ex:
        return _ex

def face_recogn():
    try:
        # result = DeepFace.find(img_path='faceBPI/img.png', db_path='faceBPI')
        # if result.empty:
        #     return "No matches found"
        # result = result.values.tolist()
        img_path = "faceBPItest/img_3.png"

        # Путь к базе данных изображений
        db_path = "faceBPItest"

        DeepFace.detectFace(img_path)


        # Выполнение поиска
        result = DeepFace.find(img_path=img_path, db_path=db_path)
        return result
    except Exception as err:
        return err

def main():
    # print(face_verify(img_1='faceBPItest/img_7.png', img_2='faceBPItest/img_6.png'))
    print(face_recogn())




if __name__ == "__main__":
    main()