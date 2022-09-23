# person_detection

Programa para detectar personas para controlar el aforo con un webcam conectada a una Raspberry Pi 3B+, con notificación por email cuando supera el umbral de aforo fijado.

Este código tiene como base el ejemlo de clasificación de imágenes del repositorio de TensorFlow Lite: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

(1)Léase el archivo de texto "LÉAME.TXT"

(2)Clone el el repositorio ofrecido por TensorFlow en la Raspberry Pi 3B+:

git clone https://github.com/tensorflow/examples --depth 1

(3) Abre el directorio:

cd examples/lite/examples/image_classification/raspberry_pi

(4) Ejecute esta linea de código para instalar en el directorio modelos de TensorFlow Lite:

sh setup.sh


Alguna ideas también fueron sacadas del código de ejemplo que proporciona la web:

https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588)
