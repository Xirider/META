import os
import time

os.system("activate qa")
time.sleep(5)
exe_str = r"cd C:/Users/sophi/Desktop/peter/qa/inference"

os.system(exe_str)

os.system("python app.py")

time.sleep(3)

os.system('start "" "http://127.0.0.1:5000/"')