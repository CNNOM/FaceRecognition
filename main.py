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



def main():
    print(face_verify(img_1='faceBPI/BPI-221_kozlovskaya.png', img_2='faceBPI/BPI-221_gamayunov.png'))



if __name__ == "__main__":
    main()