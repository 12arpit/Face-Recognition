import cv2
import numpy as np
import os

#knn portion

def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist=[]

	for i in range(train.shape[0]):
		ix=train[i, :-1]
		iy=train[1, -1]
		d=distance(test, ix)
		dist.append([id, iy])
		dk=sorted(dist, key=lambda x:x[0])[:k]

		labels=np.array(dk)[:, -1]
		output=np.unique(labels, return_counts=True)

		index=np.argmax(output[1])
		return output[0][index]

#initializing webcam
cap=cv2.VideoCapture(0)
#fdobject
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#data preparation
class_id=0
names={}
dataset_path='./data/'

face_data=[]
labels=[]

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4]

		data_item=np.load(dataset_path+fx)
		face_data.append(data_item)

		target=class_id*np.ones((data_item.shape[0],))

		class_id+=1
		labels.append(target)

face_dataset=np.concatenate(face_data, axis=0)
face_labels=np.concatenate(labels, axis=0).reshape((-1, 1))

train_set=np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

#testing

while True:
	ret, frame=cap.read()
	if ret==False:
		continue

	#conversion to grayscale
	gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces=face_cascade.detectMultiScale(gray_frame, 1.3, 5)
	if len(faces)==0:
		continue

	for face in faces:
		x,y,w,h=face

		offset=10
		face_section=gray_frame[y-offset:y+offset, x-offset:x+w+offset]
		face_section=cv2.resize(face_section, (100, 100))

		#predict
		out=knn(train_set, face_section.flatten())

		#display on screen
		pred_name=names[int(out)]
		cv2.putText(gray_frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

		cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (0,255,0), 2)

	#cv2.imshow("Frame", frame)
	cv2.imshow("gray_frame", gray_frame)

	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()