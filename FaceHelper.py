import os.path
import uuid
import requests
import argparse
import time
from PIL import Image,ExifTags
import cv2,imutils

API_KEY = "YOUR_FACE_API_KEY"
ENDPOINT = "YOUR_FACE_API_BASEURL"

def person_group_create(personGroupId):
    endpoint = ENDPOINT+'/persongroups/{}'.format(person_group_id)
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    name = personGroupId
    json = {
        'name': name
    }
    r = requests.put(endpoint, json=json,headers=headers)
    if r.status_code != 200:
        print('error:' + r.text)
    else:
        print('Person group created with ID: '+personGroupId)
def person_create(personGroupId,personName):
    endpoint = ENDPOINT+'/persongroups/{}'.format(person_group_id)+'/persons'
    print(endpoint)
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    name = personGroupId
    json = {
        'name': personName
    }
    r = requests.post(endpoint, json=json,headers=headers)
    if r.status_code != 200:
        print('error:' + r.text)
    else:
        res=r.json()
        person_id = res['personId']
        print('Person created: '+ personName+'with ID: '+person_id)
        return person_id

def person_addface(path_face, person_group_id, person_id):
    endpoint = ENDPOINT+'/persongroups/{}/persons/{}/persistedFaces'.format(person_group_id, person_id)
    headers = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/octet-stream"}
    binaryImage = resizeImagewithCV(path_face)
    r = requests.post(endpoint, headers=headers,data=binaryImage)
    if r.status_code != 200:
        print('error:' + r.text)
    else:
        res=r.json()
        if res.get('persistedFaceId'):
            persistedFaceId = res['persistedFaceId']
            print('A Face Added to the person with ID: '+person_id)
            return persistedFaceId
        else:
            print('no persistedfaceid found')

def persongroup_train(person_group_id):
    endpoint = ENDPOINT+'/persongroups/{}'.format(person_group_id)+'/train'
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    trainingstarttime=time.time()
    r=requests.post(endpoint,headers=headers)
    elapsedtime = time.time()-trainingstarttime
    if r.status_code != 202:
        print('error:' + r.text)
    else:
        print('Training is succesfully completed')
        return elapsedtime

def persongroup_list():
    endpoint = ENDPOINT+'/persongroups'
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    r=requests.get(endpoint,headers=headers)
    if r.status_code != 200:
        print('error:' + r.text)
    else:
        res=r.json()
        return res
def persongroup_delete(personGroupId):
    endpoint = ENDPOINT+'/persongroups/{}'.format(personGroupId)
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    r=requests.delete(endpoint,headers=headers)
    if r.status_code != 200:
        print('error:' + r.text)
    else:
        print('Person group with ID: '+personGroupId+'succesfully deleted')

def deletealldata():
    PGlist = persongroup_list()
    for pg in PGlist:
        persongroup_delete(pg['personGroupId'])
    print('all gone!')

def BeforeIdentification(path,person_group_id):
    person_id_names = {}
    person_name_faces = {}
    beforetrainingstart=time.time()
    person_group_create(person_group_id) #created a new person group
    for person_name in os.listdir(path):
        print(path)
        path_person = os.path.join(path, person_name)
        print(path_person)
        if os.path.isdir(path_person):
                person_id = person_create(person_group_id, person_name)
                print(person_id)
                person_id_names[person_id] = person_name
                person_name_faces[person_name] = []
                for entry in os.listdir(path_person):
                    path_face = os.path.join(path_person, entry)
                    if os.path.isfile(path_face):
                        persistedFaceId = person_addface(path_face, person_group_id, person_id)
                        person_name_faces[person_name].append(persistedFaceId)
    beforetrainingelapsed = time.time()-beforetrainingstart
    trainingelapsedtime = persongroup_train(person_group_id)
    return person_id_names,person_name_faces,trainingelapsedtime,beforetrainingelapsed

def faceDetectForIdentification(binaryImage):

    endpoint = ENDPOINT+"/detect?returnFaceId=true"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/octet-stream"}
    faceDetectForIdentificationstart = time.time()
    r = requests.post(endpoint, headers=headers, data=binaryImage)
    faceDetectForIdentificationelapsed = time.time() - faceDetectForIdentificationstart
    if r.status_code != 200:
        print(r.text)
    data = r.json()
    print(data)
    faceAreas = list(map(lambda x: x["faceRectangle"]["width"] * x["faceRectangle"]["height"], data))
    maxIndex = faceAreas.index(max(faceAreas))
    print('Face detected for identification')
    return data[maxIndex]["faceId"],faceDetectForIdentificationelapsed


def faceIdentify(binaryImage, personGroupId):

    faceId,faceDetectForIdentificationelapsed = faceDetectForIdentification(binaryImage)
    endpoint = ENDPOINT+'/identify'
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    faceIds = [faceId]

    maxNumOfCandidatesReturned = 1
    confidenceThreshold = 0.8

    json = {
        "faceIds": faceIds,
        "maxNumOfCandidatesReturned": maxNumOfCandidatesReturned,
        "personGroupId": personGroupId
    }
    faceIdentifystart=time.time()
    r = requests.post(endpoint, headers=headers, json=json)
    faceIdentifyelapsed = time.time()-faceIdentifystart
    if r.status_code != 200:
        print(r.text)
    else:
        res=r.json()
        bestcandidate = res[0]['candidates'][0]
        result_personID = bestcandidate['personId']
        result_confidence = bestcandidate['confidence']

        return result_personID,result_confidence,faceDetectForIdentificationelapsed,faceIdentifyelapsed

def resizeImagewithCV(imagepath):
	img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, None,fx=0.1,fy=0.1)
	if img.shape[1]>img.shape[0]:
		img=imutils.rotate_bound(img,angle=90)
	img_byte = cv2.imencode('.jpg', img)[1].tostring()
	return img_byte


if __name__ == '__main__':
	deletealldata()
	person_group_id=str(uuid.uuid1())
	person_id_names,person_name_faces,trainingelapsedtime,beforetrainingelapsed=BeforeIdentification('YOUR_TRAININGPHOTOS_DIRECTORY',person_group_id)
	for p in person_id_names:print('Person ID :'+p+' Person Name: '+person_id_names[p])
	testimagepath = 'YOUR_TEST_IMAGE_PATH'
	result_personID,result_confidence,faceDetectForIdentificationelapsed,faceIdentifyelapsed=faceIdentify(resizeImagewithCV(testimagepath),person_group_id)
	print('Detected person is '+str(person_id_names[result_personID])+' with ID: '+str(result_personID)+', Confidence: '+str(result_confidence))
	print('-------TIMING REPORT------')
	print('Training preparation Duration: '+str(beforetrainingelapsed))
	print('Training Duration: '+str(trainingelapsedtime))
	print('Face Detect before Identifition Duration: '+str(faceDetectForIdentificationelapsed))
	print('Face Identification Duration: '+str(faceIdentifyelapsed))
