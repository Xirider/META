import os
import time
import subprocess
os.system('start "" "http://127.0.0.1:5000/"')
subprocess.run(r"activate qa && cd C:/Users/sophi/Desktop/peter/qa/inference && python app.py", shell=True)
# os.system("activate qa")
# time.sleep(5)
# exe_str = r"cd C:/Users/sophi/Desktop/peter/qa/inference"

# os.system(exe_str)

# os.system("python app.py")


