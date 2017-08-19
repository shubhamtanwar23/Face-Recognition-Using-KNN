import numpy as np 
import cv2

def distance(x1, x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, labels, k=5):
	m= train.shape[0]
	dist = []
	for i in range(m):
		dist.append(distance(x, train[i]))
	dist = np.asarray(dist)
	indx = np.argsort(dist)
	sorted_labels = labels[indx][:k]
	counts = np.unique(sorted_labels, return_counts=True)
	return counts[0][np.argmax(counts[1])]

# instantiate the camera object and haar cascade
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# declare the font type
font = cv2.FONT_HERSHEY_SIMPLEX

# load the data from the numpy matrices and convert them into linear shape array
f_01 = np.load('face_01.npy').reshape((20, 50*50*3))
f_02 = np.load('face_02.npy').reshape((20, 50*50*3))


print(f_01.shape, f_02.shape)

# create a look up dictionary
names = { 0 : "Admin", 
		  1 : "Guest" }


# create a matrix to store labels
labels = np.zeros((40,1))
labels[:20] = 0		# Admin
labels[20:] = 1		# Guest

# combine all info into one data array
data = np.concatenate([f_01, f_02])
print (data.shape, labels.shape)

while True:
	# get each frame
	ret, frame= cam.read()

	if ret:
		# convert to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		for (x, y, w, h) in faces:
			face_component = frame[y:y+h, x:x+w, :]
			fc = cv2.resize(face_component, (50,50))

			# passing the component to knn classifier

			lab = knn(fc.flatten(), data, labels)

			# retrieve name from dictionary 
			text = names[int(lab)]

			# display the name
			cv2.putText(frame, text, (x,y), font, 1, (255,255,0), 2)

			# draw a rectangle over the face
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

		cv2.imshow('face recoginition', frame)

		if cv2.waitKey(1)==27:
			break
	else:
		print ("Camera error")

cv.destroyAllWindows()