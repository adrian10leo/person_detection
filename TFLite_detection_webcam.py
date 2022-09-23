######## Webcam para la detección de personas utilizando Tensorflow-lite #########
#
# Autor: Adrian de la Rosa
# Date: 27/07/2022

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import smtplib
import email.utils
import email
from threading import Timer
import threading as th

aforo=1        #aforo es cero cuando se supera el aforo
enviar=0         #enviar se pone a 1 cuando se manda el email


def Timer_Interrupt():
         
         th.Timer(60, email).start()
        
         
         
             

         
def email ():
     
     global enviar
     if aforo==0 and enviar==0:         #comprueba si el aforo ha bajado, si ha bajado, no se envia el email
        sender_email = "email de quien lo envía"
        rec_email = "email de quien lo recibe"
        #password = input(str("Please enter your password : "))
        password = str("contraseña del email de quien lo envía")
        message = "Subject:Informacion de Aforo\n\nAforo Completo: http://controldeaforo.site"
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(sender_email, password)
        print("Login success")
        server.sendmail(sender_email, rec_email, message)
        print("email enviado a ", rec_email)       
        enviar=1
        

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
#         self.stream = cv2.VideoCapture('sala2.mov')    #para leer desde un video
        self.stream = cv2.VideoCapture(0)    #para leer desde una webcam
#         self.stream = cv2.VideoCapture('http://192.168.1.221:8081') #para leer desde IP
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
   

   
    

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='model.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.50)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Importar TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Arreglo extraño, ya que en el mapa de etiquetas COCO
# https://www.tensorflow.org/lite/models/object_detection/overview
# La primera etiqueta es '???', la cual hay que eliminar.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# Create window
cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
contador=0

                
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # confianza de los objetos detectados
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            object_name = labels[int(classes[i])] # busca el nombre del objeto en la matriz "labels" usando la etiqueta de clase
            if object_name == 'person':
                # Get bounding box coordinates and dibujar el cuadro
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                
                # dibujar etiqueta
                
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1]+10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin +baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                # dibujar circulo rojo en el centro del marco
                xcenter = xmin +(int(round((xmax - xmin) / 2)))
                ycenter = ymin  +(int(round((ymax - ymin) / 2)))
                cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

                # Imprimir información en el terminal
                print('Object '  + str(i)  + ': ' +  object_name +  ' at (' +  str(xcenter)  + ', ' +  str(ycenter)  + ')')
                    
                #contador de personas si supera un porcentaje
                if (object_name == 'person' and scores[i] >= 0.50):
                    contador+=1
                

    #enviar email cuando supera aforo de 4 personas y ha pasado 1 min desde el último envío de email
    if (contador>=4 and aforo==1 and enviar==0):
        aforo=0
        Timer_Interrupt()
    if contador<4 and enviar==0 and aforo==0:
        aforo=1
        
                
    if enviar==1 and contador<4:
       enviar=0
                    
    #imprimo en terminal el contador de personas         
    print('Contador personas ('  + str(contador)  + ')')
    
    # Dibuja los FPS y el Aforo en la esquina del fotograma
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(1100,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    if contador>=5:
        cv2.putText(frame,'  Aforo: {0:.2f}'.format(contador),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,'  [==100%==]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),1,cv2.LINE_AA)
        cv2.putText(frame,'      STOP   ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,'  Aforo: {0:.2f}'.format(contador),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)
        if contador==0:
            cv2.putText(frame,'  [   0%   ]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),1,cv2.LINE_AA)
            cv2.putText(frame,'      GO    ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)

        if contador ==1:
            cv2.putText(frame,'  [= 20%   ]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),1,cv2.LINE_AA)
            cv2.putText(frame,'      GO    ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)
        
        if contador ==2:
            cv2.putText(frame,'  [==40%   ]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),1,cv2.LINE_AA)
            cv2.putText(frame,'      GO    ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)
            
        if contador ==3:
            cv2.putText(frame,'  [==60%=  ]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),1,cv2.LINE_AA)
            cv2.putText(frame,'      GO    ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)
            
        if contador ==4:
            cv2.putText(frame,'  [==80%== ]',(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),1,cv2.LINE_AA)
            cv2.putText(frame,'      GO    ',(30,105),cv2.FONT_HERSHEY_SIMPLEX,1,(10, 255, 0),2,cv2.LINE_AA)


    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detection', frame)

    # Calcula framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    contador=0

    # Presione 'q' para salir
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
