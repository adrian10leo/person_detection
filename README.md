# person_detection

Programa para detectar personas para controlar el aforo con un webcam conectada a una Raspberry Pi 3B+, con notificación por email cuando supera el umbral de aforo fijado.

Este código tiene como base el ejemlo de clasificación de imágenes del repositorio de TensorFlow Lite: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

Y algunas ideas tambien fueron sacadas de estos ejemplos:
https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

(1)Léase el archivo de texto "LÉAME.TXT"

(2)Clone el el repositorio ofrecido por TensorFlow en la Raspberry Pi 3B+:

git clone https://github.com/tensorflow/examples --depth 1

(3) Abre el directorio:

cd examples/lite/examples/image_classification/raspberry_pi

(4) Ejecute esta linea de código para instalar en el directorio modelos de TensorFlow Lite:

sh setup.sh


