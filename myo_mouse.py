from enum import Enum
import socket
import math
import datetime

class MyoMouse:
    def __init__(self, velocity=15, proportional_control=True):
        # Set up default values 
        self.VEL = velocity
        self.proportional_control = proportional_control

        # Socket for reading EMG
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        self.sock.bind(('127.0.0.1', 12346))
        while True:
            self.read_data()

    def read_data(self):
        import pyautogui # This is done here since it was impacting TKinter

        # Read data from the specific socket
        data, _ = self.sock.recvfrom(1024)
        data = str(data.decode("utf-8"))
        if data:
            input_class = float(data.split(' ')[0])
            multiplier = 1
            if self.proportional_control:
                multiplier = float(data.split(' ')[1])
            
            if input_class == 0:
                # Move Down
                pyautogui.moveRel(0, self.VEL * multiplier)
            elif input_class == 1:
                # Move Up
                pyautogui.moveRel(0, -self.VEL * multiplier)
            elif input_class == 3:
                # Move Right
                pyautogui.moveRel(self.VEL * multiplier, 0)
            elif input_class == 4:
                # Move Left
                pyautogui.moveRel(-self.VEL * multiplier, 0)
    