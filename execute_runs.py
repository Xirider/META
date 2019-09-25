import os
import time
import subprocess


x= "pip install slacker"
a ="python run_classifier.py --data_dir traindata/processed_d4_1_long --output_dir logfiles/d4_1_r_sampling --do_train --do_eval --bert_model finetuned_lm --gradient_accumulation_steps 16 --train_batch_size 32 --num_train_epochs 10 --overwrite_output_dir --random_sampling"

b = "python run_classifier.py --data_dir traindata/processed_d4_1_long --output_dir logfiles/d4_1_balanced_s --do_train --do_eval --bert_model finetuned_lm --gradient_accumulation_steps 16 --train_batch_size 32 --num_train_epochs 10 --overwrite_output_dir"

c = "python run_classifier.py --data_dir traindata/processed_d4_1_long --output_dir logfiles/d4_1_active_s --do_train --do_eval --bert_model finetuned_lm --gradient_accumulation_steps 16 --train_batch_size 32 --num_train_epochs 10 --overwrite_output_dir --active_sampling"


os.system(x)
os.system(a)
os.system(b)
os.system(c)
# subprocess.run(r"activate qa && cd C:/Users/sophi/Desktop/peter/qa/inference && python app.py", shell=True)

slack_token ="xoxp-67976230180-67930829795-761105709923-61d20fcbd182f4f371dbafa56b10a4b3"

from slacker import Slacker

slack = Slacker(slack_token)

# Send a message to #general channel
slack.chat.post_message('peters_secret_channel', 'Traning run finished')


# os.system("activate qa")
# time.sleep(5)
# exe_str = r"cd C:/Users/sophi/Desktop/peter/qa/inference"

# os.system(exe_str)

# os.system("python app.py")


