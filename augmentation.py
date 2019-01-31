import os, random

data_raw = './data_raw'
data_raw_list = []
for _, dirs, _ in os.walk(data_raw):
    for dir in dirs:
        if dir[0] == '.':
            continue
        data_raw_list.append(os.path.join(data_raw, dir))
# print(data_raw_list)
count_list = []
for DIR in data_raw_list:
    count_list.append(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

print(max(count_list))
max_count = max(count_list)

op_list = ['fliph', 'flipv', 'noise_0.01', 'noise_0.5', \
           'rot_90', 'rot_-45', 'trans_20_10', 'trans_-10_0',\
           'zoom_0_0_20_20', 'zoom_-10_-20_10_10', 'blur_1.5']

for dir in data_raw_list:
    print(dir)
    while len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]) < max_count:
        os.system('python3 main.py {} {} {}'.format(dir, random.choice(op_list), random.choice(op_list)))
        # print(dir, random.choice(op_list), random.choice(op_list))
        count = 0
        for _, _, files in os.walk(dir):
            for file in files:
                os.rename(os.path.join(dir,file), os.path.join(dir, dir.split('/')[-1]+'_'+str(count)+".jpg"))
                count += 1
