
import cv2
import numpy as np
import face_recognition
import os
import time


#automatically get people in database
path='people'
images=[]
classNames=[]
mylist= os.listdir(path)
print(mylist)

#append the names of images to classes
for cn in mylist:
    cur_img=cv2.imread(f'{path}/{cn}')
    images.append(cur_img)
    classNames.append(os.path.splitext(cn)[0])
print(classNames)

#create a function that will get encodings
def getEncodings (images):
    encodeList=[]
    for img in images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

myknownList= getEncodings(images)
print(len(myknownList))

#make process with web cam
cap=  cv2.VideoCapture(0)

while True:
    success , img =cap.read()
    # decrease size of img to quarter
    imgSml= cv2.resize(img,(0,0),None,0.25,0.25)
    imgSml = cv2.cvtColor(imgSml,cv2.COLOR_BGR2RGB)

    facesinCurFrame= face_recognition.face_locations(imgSml)
    encodinCurFram=face_recognition.face_encodings(imgSml,facesinCurFrame)

    for encodeFace,Facloc in zip (encodinCurFram,facesinCurFrame):
        matches = face_recognition.compare_faces(myknownList,encodeFace)
        faceDistance = face_recognition.face_distance(myknownList,encodeFace)
        print(faceDistance)

        #get the index of the class by getting the minimal distance
        #min distance is least error
        matchIndex = np.argmin(faceDistance)
        time.sleep(1)
        if matches[matchIndex]:
            name= classNames[matchIndex]
            print(name)
        if faceDistance[matchIndex]>=0.5:
            print("intruder")

            time.sleep(10)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)