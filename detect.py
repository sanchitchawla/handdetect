import cv2
import numpy as np

#Open the camera
cap = cv2.VideoCapture(0)

def passing(x):
	pass

#Set frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Reusable function to find angle between two vectors
def angle(v1,v2):
	dot = np.dot(v1,v2)
	x_mod = np.sqrt((v1*v1).sum())
	y_mod = np.sqrt((v2*v2).sum())
	cos_angle = dot / x_mod / y_mod
	return np.degrees(np.arcos(cos_angle))

# Reusable function to find the distance between two points in a list of lists
def findDist(X, Y):
	return np.sqrt((X[0][0] - Y[0][0])**2 + (X[0][1] - Y[0][1])**2)

# Create a window for HSV track bars
cv2.namedWindow("HSV_track")

h,s,v = 100,100,100

# Create Trackbars
cv2.createTrackbar('h', 'HSV_track', 0, 179, passing)
cv2.createTrackbar('s', 'HSV_track', 0, 255, passing)
cv2.createTrackbar('v', 'HSV_track', 0, 255, passing)

while (True):

	# Capture frames from the camera
	ret, frame = cap.read()

	# Blur the image
	blurred = cv2.blur(frame, (3,3))

	# Convert to HSV colour space
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# Create a binary image. white = skin colour and rest is black
	mask2 = cv2.inRange(hsv, np.array([2,50,50]), np.array([15,255,255]))

	kernal_square = np.ones((11,11), np.uint8)
	kernal_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

	# Morphological transformation to filter background noise

	dilation = cv2.dilate(mask2, kernal_ellipse, iterations = 1)
	erosion  = cv2.erode(dilation, kernal_square,iterations = 1)
	dilation2 = cv2.dilate(erosion, kernal_ellipse, iterations = 1)
	filtered  = cv2.medianBlur(dilation2, 5)
	kernal_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilation2 = cv2.dilate(filtered, kernal_ellipse, iterations = 1)
	kernal_ellipse = cv2.dilate(filtered, kernal_ellipse, iterations = 1)
	median = cv2.medianBlur(dilation2, 5)
	ret, threshold = cv2.threshold(median, 127,255,0)

	# Get contours 
	_, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Find max contour area

	max_area = 100
	ci = 0 # contour index

	for i in range(len(contours)):
		cnt = contours[i]
		area= cv2.contourArea(cnt)
		if (area > max_area):
			max_area = area
			ci = i


	cnts = contours[ci] # Largest
	hull = cv2.convexHull(cnts)

	# Find convex defects
	hull2 = cv2.convexHull(cnts, returnPoints = False)
	defects = cv2.convexityDefects(cnts, hull2)

	# Get defect points and draw them in the original image
	AllDefect = []
	for i in range(defects.shape[0]):
	    s,e,f,d = defects[i,0]
	    start = tuple(cnts[s][0])
	    end = tuple(cnts[e][0])
	    far = tuple(cnts[f][0])
	    AllDefect.append(far)
	    cv2.line(frame,start,end,[0,255,0],1)
	    cv2.circle(frame,far,10,[100,255,255],3)

	moments = cv2.moments(cnts)

	# Central mass of first order moments
	if moments['m00']!=0:
	    cx = int(moments['m10']/moments['m00'])
	    cy = int(moments['m01']/moments['m00'])
	centerMass=(cx,cy)    

	# Draw center mass
	cv2.circle(frame,centerMass,7,[100,0,255],2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)  

	# Distance from each finger defect to the center mass
	distanceBetweenDefectsToCenter = []
	for i in range(0,len(AllDefect)):
	    x =  np.array(AllDefect[i])
	    centerMass = np.array(centerMass)
	    distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
	    distanceBetweenDefectsToCenter.append(distance)

	sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
	AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

	# Get fingertips, if points are in proximity of 80 pixels, it's a single point in a group
	finger = []

	for i in range(0, len(hull) - 1):
		if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80:
			if (hull[i][0][1] < 500):
				finger.append(hull[i][0])

	# The fingertip points are 5 hull points with largest y coordinates
	finger =  sorted(finger,key=lambda x: x[1])   
	fingers = finger[0:5]

	# Calculate distance of each finger tip to the center mass
	fingerDistance = []
	for i in range(0,len(fingers)):
	    distance = np.sqrt((fingers[i][0]-centerMass[0])**2 + (fingers[i][1]-centerMass[0])**2)
	    fingerDistance.append(distance)

	# Finger is raised if distance between finger and center of mass is > 130
	result = 0
	for i in range(0, len(fingers)):
		if fingerDistance[i] > AverageDefectDistance + 130:
			result+=1

	# Print number of pointed fingers
	cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)

	# Print bounding rectangle 
	x,y,w,h = cv2.boundingRect(cnts)
	img = cv2.rectangle(frame, (x,y), (x+w, y+ h), (0,255,0), 2)


	cv2.drawContours(frame,[hull],-1,(255,255,255),2)
	cv2.imshow('Dilation',frame)
	#close the output video by pressing 'ESC'
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
	    break


cap.release()
cv2.destroyAllWindows()
