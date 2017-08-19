import cv2
import numpy as np

# instantiate a camera object 
cam = cv2.VideoCapture(0)


# create a haar-cascade object for face detection 
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create a list for storing the data
data = []
ix = 0		# frame number

while True:

	# retrieve ret anf frame from camera
	ret, frame = cam.read()

	# if the camera is working fine we procees to extract the face
	if ret == True:

		# convert the frame to grayscale 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# apply the haar cascade to detect faces in the current frame
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		# for each face object we get corner coords (x,y) and width and height
		for (x, y, w, h) in faces : 

			# get face component
			face_component = frame[y:y+h, x:x+w, :]

			# resize the face to 50X50X3
			fc = cv2.resize(face_component, (50,50))

			# store the face data after every 10 frames only if we have less than 20 enteries
			if ix%10 == 0 and len(data)<20:
				data.append(fc)

			# for visualization draw rectangle around face in image
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

		ix += 1
		cv2.imshow('frame', frame)

		# if user press escape key(27) or the number of images are 20 then stop recording
		if cv2.waitKey(1) == 27 or len(data) >= 20:
			break
	else:
		print('error camera not functioning')

cv2.destroyAllWindows()

# convert data into numpy array
data = np.asarray(data)

print (data.shape)

np.save('face_02', data)

