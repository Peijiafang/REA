import os 


path = "workspace/SegDetectorModel-td500/deformable_resnet50/BiFPN_1layer_1_1.5/L1BalanceCELoss/model"
file_list = os.listdir(path)
if 'final' in file_list:
    file_list.remove('final')
with open('text.txt', 'w') as f:
    for files in sorted(file_list, key=lambda x: int(x.split('_')[-1])):
        f.write(files + '\n')
