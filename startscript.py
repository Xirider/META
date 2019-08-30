import subprocess


subprocess.call("activate qa")
exe_str = r"cd C:/Users/sophi/Desktop/peter/qa/inference"

subprocess.call(exe_str)

subprocess.call("python app.py")
