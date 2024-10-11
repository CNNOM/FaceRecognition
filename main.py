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
        img_path = "faceBPItest/img_3.png"
        db_path = "faceBPI"

        DeepFace.build_model("Facenet")

        result = DeepFace.find(img_path=img_path, db_path=db_path)
        return result
    except Exception as err:
        return err


def face_analyze():
    try:
        # result_dict = DeepFace.analyze(img_path='faceBPI/img.png', actions=['age', 'gender', 'race', 'emotion'])
        result_list = DeepFace.analyze(img_path='faceBPI/img.png', actions=['age', 'gender', 'race'])

        for i, result_dict in enumerate(result_list):
            print(f'[+] Лицо {i + 1}:')
            print(f'[+] Возраст: {result_dict.get("age")}')
            print(f'[+] Пол: {result_dict.get("gender")}')

            print('[+] Раса:')
            for k, v in result_dict.get('race').items():
                print(f'{k} - {round(v, 2)}%')

        # print('[+] Emotions:')
        # for k, v in result_dict.get('emotion').items(): print(f'{k} - {round(v, 2)}*')

        with open('face_analyze.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        # return result_dict
    except Exception as err:
        return err

def main():
    # print(face_verify(img_1='faceBPItest/img_7.png', img_2='faceBPItest/img_6.png'))
    # print(face_recogn())
    print(face_analyze())




if __name__ == "__main__":
    main()