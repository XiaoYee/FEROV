import os
import cPickle

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# if os.path.isdir('TrainTestlist') is False:
# 	os.mkdir('TrainTestlist')

file_class = open(os.path.join('/home/quxiaoye/MMI_Faces','TrainTestlist', 'classInd.txt'), 'w')
file_train = open(os.path.join('/home/quxiaoye/MMI_Faces','TrainTestlist', 'trainlist.txt'), 'w')
file_val = open(os.path.join('/home/quxiaoye/MMI_Faces','TrainTestlist', 'vallist.txt'), 'w')
info = {}

i = 0
for item in emotion:
	file_class.write(str(i+1) + ' ' + item + '\n')
	# train file
	train_dir = os.walk(os.path.join('/home/quxiaoye/MMI_Faces','Train_One', item))
	for root, dirs, files in train_dir:
		for d in dirs:
			print os.path.join('/home/quxiaoye/MMI_Faces/Train_One', item, d)
			file_train.write(os.path.join('Train_One', item, d) + '\n')
			info[os.path.join('/home/quxiaoye/MMI_Faces/Train_One', item, d)] = 0
	# panasonic
	# train_dir = os.walk(os.path.join('panasonic-faces', item))
	# for root, dirs, files in train_dir:
	# 	for d in dirs:
	# 		print os.path.join('panasonic-faces', item, d)
	# 		file_train.write(os.path.join('panasonic-faces', item, d) + '\n')
	# 		info[os.path.join('panasonic-faces', item, d)] = 0

	# # CK
	# train_dir = os.walk(os.path.join('CK-faces', item))
	# for root, dirs, files in train_dir:
	# 	for d in dirs:
	# 		print os.path.join('CK-faces', item, d)
	# 		file_train.write(os.path.join('CK-faces', item, d) + '\n')
	# 		info[os.path.join('CK-faces', item, d)] = 0
	# val file
	val_dir = os.walk(os.path.join('/home/quxiaoye/MMI_Faces','Test_One', item))
	for root, dirs, files in val_dir:
		for d in dirs:
			print os.path.join('/home/quxiaoye/MMI_Faces/Test_One', item, d)
			file_val.write(os.path.join('Test_One', item, d)+ '\n')
			info[os.path.join('/home/quxiaoye/MMI_Faces/Test_One', item, d)] = 0
	i += 1

file_class.close()
file_train.close()
file_val.close()

for video in info:
	list_dir = os.walk(video)
	count = 0
	for root, dirs, files in list_dir:
		for f in files:
			#print os.path.join(root, f)
			count += 1
	info[video] = count
file = open(os.path.join('info.pkl'),'wb')
cPickle.dump(info, file)