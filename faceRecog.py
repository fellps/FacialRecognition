#!/usr/bin/python
from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import os
import os.path
import pickle

# Mostrar as características faciais
show_face_landmarks = True
# Mostrar caixa ao redor do rosto
show_face_box = True
# Somente retornar o valor para o script que está chamando
return_value = False

known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_landmarks = []
face_names = []
process_this_frame = True
face_encodings_config = 'known_face_encodings.config'
face_names_config = 'known_face_names.config'
train_directory = 'images/'
font = cv2.FONT_HERSHEY_DUPLEX

print("Iniciando treinamento..")

if not os.path.isfile(face_encodings_config):
    # Diretório de treino
    train_dir = os.listdir(train_directory)

    # Percorrer cada pessoa no diretório de treinamento
    for person in train_dir:
        pix = os.listdir(train_directory + person)

        # Percorrer cada imagem de treinamento da pessoa atual
        for person_img in pix:
            # Obtem as codificações de rosto em cada arquivo de imagem
            face = face_recognition.load_image_file(train_directory + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # Se a imagem de treinamento conter exatamente uma face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Adiciona a codificação da face para a imagem atual com (nome e cpf) correspondente aos dados de treinamento
                known_face_encodings.append(face_enc)
                known_face_names.append(person)
                print("Importação realizada: " + person_img)
            else:
                print(person + "/" + person_img + " nao pode ser usada para o treino!")
    
    # Grava as codificações de rosto em um arquivo
    with open(face_encodings_config, "wb") as fp:
        pickle.dump(known_face_encodings, fp)

    # Grava os nomes associados as condificações de rosto
    with open(face_names_config, "wb") as fp:
        pickle.dump(known_face_names, fp)
else:
    # Recupera os dados de treinamento dos arquivos
    with open(face_encodings_config, "rb") as fp:
        known_face_encodings = pickle.load(fp)
    with open(face_names_config, "rb") as fp:
        known_face_names = pickle.load(fp)

print("Treinamento finalizado!")

print("Iniciando captura de vídeo..")
video_capture = cv2.VideoCapture(0, return_value and cv2.CAP_DSHOW or cv2.CAP_ANY)

while True:
    # Obtem um único quadro de vídeo
    ret, frame = video_capture.read()

    # Redimensiona o quadro do vídeo para 1/4 de tamanho, para o processamento de reconhecimento de rosto mais rápido
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converte a imagem de BGR (que o OpenCV usa) para RGB (que o face_recognition usa)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Processe apenas todos os outros quadros de vídeo para economizar tempo
    if process_this_frame:
        # Encontra todas as faces e codificações de face no quadro atual do vídeo
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_cpf = []
        for face_encoding in face_encodings:
            # Verifica se o rosto corresponde a um rosto conhecido
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Anonimo"
            cpf = str()

            # # Se uma correspondência foi encontrada em known_face_encodings, use a primeira.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Ou então, use a face conhecida com a menor distância para a nova face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index].split('-')[0]
                cpf = known_face_names[best_match_index].split('-')[1]
                # Usuário encontrado
                print(known_face_names[best_match_index])

            face_names.append(name)
            face_cpf.append(cpf)

    if return_value and len(face_names) > 0:
        break

    process_this_frame = not process_this_frame

    if not return_value:
        # Obtem as característica faciais (olho, nariz, boca, lábios, etc)
        if show_face_landmarks:
            face_landmarks_list = face_recognition.face_landmarks(frame)

            # Loop sobre cada característica facial (eye, nose, mouth, lips, etc)
            for face_landmarks in face_landmarks_list:
                for name, list_of_points in face_landmarks.items():
                    hull = np.array(face_landmarks[name])
                    hull_landmark = cv2.convexHull(hull)
                    cv2.drawContours(frame, hull_landmark, -1, (0, 255, 0), 2)

        # Mostra os resultados
        if show_face_box:
            for (top, right, bottom, left), name, cpf in zip(face_locations, face_names, face_cpf):
                # Escala os locais das faces, pois o quadro em que detectamos foi dimensionado para 1/4 do tamanho
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Desenha uma caixa ao redor do rosto
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
                # Desenha um rótulo com um nome abaixo da face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Desenha o cpf abaixo do rótulo
                cv2.putText(frame, cpf, (left + 6, bottom + 20), font, 0.5, (0, 0, 255), 1)

        # Mostra o fps da camera
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, "FPS: {0}".format(fps), (10, 25), font, 0.5, (255, 255, 255), 1)

        # Exibe a imagem resultante
        cv2.imshow('Video', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza a webcam
try:
    video_capture.release()
    cv2.destroyAllWindows()
except:
    print()
