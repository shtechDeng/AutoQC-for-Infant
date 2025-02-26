import random

data = []
with open("data_all_pass_fail10_12.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip("\n")
        data.append(line)

print(data)
data_len = len(data)
train = []
test = []
random.shuffle(data)
length = len(data)
count0 = 0
count1 = 0
test_len = length*0.2
train_len = length*0.8
for i in range(data_len):
    _, label = data[i].split(" ")
    label = label.strip(".0")
    if label == "2" and count1 <= test_len/2:
        count1 += 1
        test.append(data[i])
        data[i] = -1



for i in range(data_len):
    if data[i] != -1:
        _, label = data[i].split(" ")
        if label == "0" and count0 <= test_len/2:
            count0 += 1
            test.append(data[i])
            data[i] = -1


for i in range(len(data)):
    if data[i] != -1:
        train.append(data[i])


with open("train0.txt","w") as tf:
    for t in train:
        tf.write(t+"\n")

with open("test0.txt","w") as tsf:
    for t in test:
        tsf.write(t+"\n")