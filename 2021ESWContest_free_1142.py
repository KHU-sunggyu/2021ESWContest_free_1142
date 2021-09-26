# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import math
import RPi.GPIO as GPIO
from time import sleep
import os.path
from twilio.rest import Client


import serial

servoPin          = 12   # 서보 핀
SERVO_MAX_DUTY    = 12   # 서보의 최대(180도) 위치의 주기
SERVO_MIN_DUTY    = 3    # 서보의 최소(0도) 위치의 주기
servoPin2 = 32 #pwm


account_sid = 'AC64796e5732f784fef2fe311ff64117ed'
auth_token = 'bbd372ceef3555bc02dd5cc048ec340f'
client = Client(account_sid, auth_token)

GPIO.setmode(GPIO.BOARD)        # GPIO 설정
GPIO.setup(servoPin, GPIO.OUT)  # 서보핀 출력으로 설정
GPIO.setup(servoPin2, GPIO.OUT)

servo = GPIO.PWM(servoPin, 50)  # 서보핀을 PWM 모드 50Hz로 사용하기 (50Hz > 20ms)
servo2= GPIO.PWM(servoPin2, 50)

ser = serial.Serial("/dev/ttyS0", 115200, timeout=0)

STOP= 0
FORWARD = 1
BACKWORD =2

CH1=0
CH2=1

OUTPUT=1
INPUT =0
HIGH=1
LOW =0


#GPIO PIN
ENA = 33  
ENB = 33   
IN1 = 37 
IN2 = 35 
IN3 = 31 
IN4 = 29   

# 핀 설정 함수
GPIO.setup(37, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)
GPIO.setup(33, GPIO.OUT)
GPIO.setup(31, GPIO.OUT)
GPIO.setup(29, GPIO.OUT)

pwm = GPIO.PWM(33,100)


GPIO.setup(18, GPIO.OUT)


class AI:
    def __init__(self):
        self.x=0
        self.counting = 0
        self.sta=0
        self.active=1
        self.classes = []
        self.center_x =0
        self.center_y =0
        self.frame = np.empty(shape=(480,640,3))
        self.new_frame = np.empty(shape=(416,416,3))
        self.capture = np.empty(shape=(416,416,3))
        self.start_flag = False
        self.show = False
        self.YOLO_net = cv2.dnn.readNet("yolov3-custom_final.weights", "yolov3-custom.cfg")
        self.layer_names = self.YOLO_net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.YOLO_net.getUnconnectedOutLayers()]
    
    def GetName(self):
        # YOLO NETWORK 재구성
        self.classes = []
        with open("yolov3.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def Detection(self):
        start_while = True
        while (start_while is True):
            if(self.active == 1 and self.start_flag is True):
                lock = threading.Lock()
                lock.acquire()
                frame_show = self.frame
                h, w, c = frame_show.shape
                blob = cv2.dnn.blobFromImage(frame_show, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.YOLO_net.setInput(blob)
                outs = self.YOLO_net.forward(self.output_layers)

                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:

                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            # Object detected
                            self.center_x = int(detection[0] * w)
                            self.center_y = int(detection[1] * h)
                            dw = int(detection[2] * w)
                            print("in function:", self.center_x)
                            dh = int(detection[3] * h)
                            # Rectangle coordinate
                            x = int(self.center_x - dw / 2)
                            y = int(self.center_y - dh / 2)
                            boxes.append([x, y, dw, dh])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                            if(self.active):
                                self.active=0
                                self.detection_state = True
                                self.sta=1 

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(self.classes[class_ids[i]])
                        score = confidences[i]
                        
                        # 경계상자와 클래스 정보 이미지에 입력
                        cv2.rectangle(frame_show, (x, y), (x + w, y + h), (0, 0, 255), 5)
                        cv2.putText(frame_show, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, 
                        (255, 255, 255), 1)

                if  self.active ==0 :
                    start_while = False
                lock.release()
            else:
                print("Not Start")
                
        self.new_frame = frame_show
        self.show = True

    def Camera(self):
        capture=1 #FLAG
        VideoSignal = cv2.VideoCapture(-1)
        self.GetName()
        self.threadControl()
        while (True):
            ret, self.frame = VideoSignal.read()
            
           
            if(self.start_flag is False):
                self.start_flag = True
                print(self.start_flag)
            else:
                pass

            if(self.sta and self.show is True):
                if(capture): 
                    self.capture = self.new_frame
                    capture=0
                    cv2.imwrite('Detection_capture.jpg',self.capture, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
                    
                if(self.active ==0):
                    
                   break
                

class SubControl(AI):
    def __init__(self):
         AI.__init__(self)
         

         self.degree = 0
         self.distance_x = 0
         self.after_x=0
         self.distance_y = 0
         self.degree_y = 0
         self.length=0
         self.finish = True

         #check if detection is success
         self.detection_state = False
         self.checking = False
         self.count = 0
         self.num = 10
         self.distance= []
         self.on_TFmini = True
         self.add_degree =0
         self.r = 0
         self.direction = 0

     
         
        
    def cal_x(self):
       
        print("in cal_x:", self.center_x)
        
        self.distance_x = (abs(self.center_x-319))*(0.0002645833)
      
        p = 20
        result = self.distance_x/p
        print("self.distance", self.distance_x)
        print("result:", result)
        a = math.asin(result)
        print(a) 
        self.degree= (a*180*180/3.14159/3.14159)
       
        print("degree",self.degree)
        
    def setServoPos(self):
        
        if self.degree >180:
            self.degree = 180
        print("degree",self.degree)
        if(self.center_x >=319):
            duty = (SERVO_MIN_DUTY+((self.num-self.degree-3)*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0))
        else:
            duty = (SERVO_MIN_DUTY+((self.num+self.degree-3)*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0))
            
        print("duty:",duty)
        
        servo.ChangeDutyCycle(duty)
        sleep(1)
        servo.stop()
        
        VideoSignal = cv2.VideoCapture(-1)
        ret, frame = VideoSignal.read()
        sleep(1.5)
        cv2.imwrite("after.jpg", frame, params= [cv2.IMWRITE_PNG_COMPRESSION,0])
        print(duty)

    def move_motor_degree(self, num):
        
        print("[Rotate degree] : ", num)
        
        duty = (SERVO_MIN_DUTY+(num*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0))
        servo.ChangeDutyCycle(duty)
        sleep(8)
      
    def move_motor_degree_y(self, num):
      
        sleep(1)
        
        print("move_motor_degree_y")
        duty = (SERVO_MIN_DUTY+(num*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0))
        servo2.ChangeDutyCycle(duty)
        sleep(2)
    
    def convert_distance(self):
        x = 0
        self.finish = True 
        
        while(self.finish):
            if(x <= self.r < x+3):
                self.add_degree = int(x/6.67)
                self.finish =False
                print(self.r)
                print("get in if")
                
            elif(x== 277):
                self.finish = False
                print("x==277")
            else:
                x+=3
        print(self.r)
    

    
    def setMotorContorl(self,pwm, INA, INB, speed, stat):
    
        #모터 속도 제어 PWM
        pwm.ChangeDutyCycle(speed)  
        
        if stat == FORWARD:
            GPIO.output(INA, HIGH)
            GPIO.output(INB, LOW)
            
        #뒤로
        elif stat == BACKWORD:
            GPIO.output(INA, LOW)
            GPIO.output(INB, HIGH)
            
        #정지
        elif stat == STOP:
            GPIO.output(INA, LOW)
            GPIO.output(INB, LOW)
    
    def setMotor(self, ch, speed, stat):
        if ch == CH1:
           
            self.setMotorContorl(pwm, IN1, IN2, speed, stat)
        else:
        
            self.setMotorContorl(pwm, IN3, IN4, speed, stat)
  

    def check_detection_success(self):
        
        if self.count == 0:
            servo.start(0)
            servo2.start(0)
            self.move_motor_degree(0)
            print("[Init]")
            self.count +=1
        elif(self.detection_state is False and self.count < 20):
            self.move_motor_degree(self.num)
            self.count += 1
            self.num +=10
        elif(self.detection_state is True and self.count < 20):
            message = client.messages.create(
                body = "Help, Help! Call 119 please",
                from_= "+19286156838",
                to = "+821038836709"
            )
            
            self.cal_x()
            print("self.num:", self.num)
            self.setServoPos()
            sleep(1.5)
            self.on_TFmini = True
            self.checking = True
            self.sensor()
            if(self.on_TFmini is False):
                self.r = self.distance[self.length-1] 
                self.move_motor_degree_y(0)
                sleep(2)
                self.convert_distance()
                print("y degree:",self.add_degree)
                trial = int(self.add_degree/10)
                if(trial ==0 or trial == 1):
                    self.move_motor_degree_y(self.add_degree)
                    sleep(0.5)
                elif(trial ==2):
                    self.move_motor_degree_y(10)
                    sleep(0.3)
                    self.move_motor_degree_y(self.add_degree)
                    sleep(0.3)
                elif(trial==3):
                    self.move_motor_degree_y(10)
                    sleep(0.3)
                    self.move_motor_degree_y(20)
                    sleep(0.3)
                    self.move_motor_degree_y(self.add_degree)
                    sleep(0.3)
                else:
                    self.move_motor_degree_y(10)
                    sleep(0.3)
                    self.move_motor_degree_y(20)
                    sleep(0.3)
                    self.move_motor_degree_y(30)
                    sleep(0.3)
                    self.move_motor_degree_y(self.add_degree)
                    sleep(0.3)
                
                servo2.stop()
                pwm.start(0)
                sleep(0.1)
        
                file = 'input.txt'
                if(os.path.isfile(file)):
                    print('file exist','motor start')
                 
                    self.setMotor(CH1, 100, FORWARD)
                    self.setMotor(CH2, 100, FORWARD)
                    sleep(2)
                    pwm.ChangeDutyCycle(3)
                    sleep(0.6)
                    self.setMotor(CH1, 100, FORWARD)
                    self.setMotor(CH2, 100, FORWARD)
                    sleep(3)
                    self.setMotor(CH1, 100, STOP)
                    self.setMotor(CH2, 100, STOP)
                    
                    pwm.stop()
                    
                    GPIO.cleanup()
                    self.active = 1
                else:
                    self.active =1
                    
        
                    
        elif(self.count > 19):
            print("[Return to Initial Pose]")
            self.move_motor_degree(140)
            sleep(0.1)
            self.move_motor_degree(90)
            sleep(0.1)
            self.move_motor_degree(45)
            sleep(0.1)
            self.move_motor_degree(0)
            self.count = 0
            self.num = 10

    def check_detection_state(self, time):
        while(self.checking is False):
            if(self.start_flag is True):
                self.check_detection_success()
                sleep(time)
            else:
                continue

    def threadControl(self):
        detection_thread = threading.Thread(target=self.Detection)
        check_detection_success_thread = threading.Thread(target=self.check_detection_state, args=(1,))
        detection_thread.start()
        check_detection_success_thread.start()
    
    def getTFminiData(self):
        while (self.on_TFmini):
            count = ser.in_waiting
            if count>20:
                recv = ser.read(9)
                ser.reset_input_buffer()
                if(recv[0]=='Y' and recv[1]=='Y'):
                    low = int(recv[2].encode('hex'), 16)
                    high = int(recv[3].encode('hex'), 16)
                    self.x = low+high*256
                    print(self.x)
             
                else:
                    ser.reset_input_buffer()
                    sleep(0.005)
            else:
                sleep(0.005)
            self.distance.append(self.x)
            self.length = len(self.distance)
         
            
            if(abs(self.distance[self.length-1]- self.distance[self.length-2]) <=7):
                self.counting +=1
                if(self.counting == 200):
                    self.counting = 0
                    self.on_TFmini = False
            
                
    def sensor(self):
        if ser.is_open == False:
            ser.open()
        self.getTFminiData()


    def main(self):
        print("[Start]")
        self.Camera()
        print("[End]")

    

    

if __name__ == "__main__":  

    subControl = SubControl()
    subControl.main()
   




