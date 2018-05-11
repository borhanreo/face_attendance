import threading
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
TRIG = 19
ECHO = 26
pulse_end_01 =0.00
pulse_start_01 = 0.00
pulse_duration =0.00
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
sensor_enable=True


def read_sensor():
    fleft=False
    while True:
        print("borhan")
        print ("dfsfdsf")
        GPIO.output(TRIG, False)
        time.sleep(0.1)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        GPIO.setwarnings(False)

        while GPIO.input(ECHO) == 0:
            global pulse_start_01
            pulse_start_01 = time.time()
            # print ("Start ", pulse_start)
        while GPIO.input(ECHO) == 1:
            global pulse_end_01
            pulse_end_01 = time.time()

        pulse_duration = pulse_end_01 - pulse_start_01
        distance = pulse_duration * 17150
        distance = round(distance, 2)

        if distance > 10 and distance < 400:
            if (distance < 100):
                print ("de")
                # global obsCounter
                # obsCounter=obsCounter+1
                # object_dectec()
                # obs_test()
            # print ("Distance:", distance - 0.5, "cm", obsCounter," ", modeValue)
            print ("Distance:", distance - 0.5, "cm", " ")
        else:
            if (fleft):
                fleft = False
thread = threading.Thread(target=read_sensor)
thread.daemon = True
thread.start()
