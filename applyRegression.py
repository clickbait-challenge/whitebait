#Script for internal evaluation of different models
import subprocess

import pickle
import numpy as np
import json
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from keras.models import model_from_yaml

### Fields in instances.jsonl:
class Data:
    def __init__(self, id=None, postTimestamp=None, postText=None, postMedia=None, targetTitle=None, targetDescription=None, targetKeywords=None, targetParagraphs=None, targetCaptions=None, groundTruth=None):
        self._id = id                                   #"<instance id>",
        self._postTimestamp = postTimestamp             #"<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",
        self._postText = postText                       # ["<text of the post with links removed>"],
        self._postMedia = postMedia                     #["<path to a file in the media archive>"],
        self._targetTitle = targetTitle                 # <title of target article>",
        self._targetDescription = targetDescription     #"<description tag of target article>",
        self._targetKeywords = targetKeywords           # "<keywords tag of target article>",
        self._targetParagraphs = targetParagraphs       #["<text of the ith paragraph in the target article>"],
        self._targetCaptions = targetCaptions           #["<caption of the ith image in the target article>"]

        self._groundTruth = groundTruth                 # Groundtruth, when applicable and provided in second file

        @property
        def id(self):
            return self._id

        @property
        def postTimestamp(self):
            return self._postTimestamp

        @property
        def postText(self):
            return self._postText

        @property
        def postMedia(self):
            return self._postMedia

        @property
        def targetTitle(self):
            return self._targetTitle

        @property
        def targetDescription(self):
            return self._targetDescription

        @property
        def targetKeywords(self):
            return self._targetKeywords

        @property
        def targetParagraphs(self):
            return self._targetParagraphs

        @property
        def targetCaptions(self):
            return self._targetCaptions

        @property
        def groundTruth(self):
            return self._groundTruth


        @id.setter
        def id(self, value):
            self._id = value

        @postTimestamp.setter
        def postTimestamp(self, value):
            self._postTimestamp = value

        @postText.setter
        def postText(self, value):
            self._postText = value

        @postMedia.setter
        def postMedia(self, value):
            self._postMedia = value

        @targetTitle.setter
        def targetTitle(self, value):
            self._targetTitle = value

        @targetDescription.setter
        def targetDescription(self, value):
            self._targetDescription = value

        @targetKeywords.setter
        def targetKeywords(self, value):
            self._targetKeywords = value

        @targetParagraphs.setter
        def targetParagraphs(self, value):
            self._targetParagraphs = value

        @targetCaptions.setter
        def targetCaptions(self, value):
            self._targetCaptions = value

        @groundTruth.setter
        def groundTruth(self, value):
            self._groundTruth = value

class GroundTruth:
    def __init__(self, id=None, truthJudgments =None, truthMean =None, truthMedian=None, truthMode=None, truthClass=None):
        self._id = id                           # "<instance id>",
        self._truthJudgments = truthJudgments   # [<number in [0,1]>],
        self._truthMean = truthMean             # <number in [0,1]>,
        self._truthMedian = truthMedian         # <number in [0,1]>,
        self._truthMode = truthMode             # <number in [0,1]>,
        self._truthClass = truthClass           # "clickbait | no-clickbait"

        @property
        def id(self):
            return self._id

        @property
        def truthJudgments(self):
            return self._truthJudgments

        @property
        def truthMean(self):
            return self._truthMean

        @property
        def truthMedian(self):
            return self._truthMedian

        @property
        def truthMode(self):
            return self._truthMode

        @property
        def truthClass(self):
            return self._truthClass

        @id.setter
        def id(self, value):
            self._id = value

        @truthJudgments.setter
        def truthJudgments(self, value):
            self._truthJudgments = value

        @truthMean.setter
        def truthMean(self, value):
            self._truthMean = value

        @truthMedian.setter
        def truthMedian(self, value):
            self._truthMedian = value

        @truthMode.setter
        def truthMode(self, value):
            self._truthMode = value

        @truthClass.setter
        def truthClass(self, value):
            self._truthClass = value

def parseData( line ):
    inst = json.loads(line)
    instance = Data(id=inst['id'], postTimestamp=inst['postTimestamp'], postText=inst['postText'], postMedia=inst['postMedia'], targetTitle=inst['targetTitle'],
                    targetDescription=inst['targetDescription'], targetKeywords=inst['targetKeywords'], targetParagraphs=inst['targetParagraphs'], targetCaptions=inst['targetCaptions'])
    return instance

def parseGT( line ):
    inst = json.loads(line)
    instance = GroundTruth(id=inst['id'], truthJudgments =inst['truthJudgments'], truthMean =inst['truthMean'], truthMedian=inst['truthMedian'], truthMode=inst['truthMode'], truthClass=inst['truthClass'])
    return instance

#1.) Load relevant processing data
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630//binary/processors.obj",'rb')
postTextTokenizer, targetTitleTokenizer, targetDescriptionTokenizer, targetKeywordsTokenizer, targetParagraphTokenizer, dayEncoder, hourEncoder, labelEncoder, classes, trainValues, colnames, trainNumbers = pickle.load(file)

file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630//binary/vars.obj",'rb')
MAX_POST_TEXT_LENGTH, MAX_TARGET_TITLE_LENGTH, MAX_TARGET_DESCRIPTION_LENGTH, MAX_TARGET_KEYWORDS_LENGTH, MAX_TARGET_PARAGRAPH_LENGTH = pickle.load(file)


file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/processors.obj",'rb')
postTextTokenizer, targetTitleTokenizer, targetDescriptionTokenizer, targetKeywordsTokenizer, targetParagraphTokenizer, dayEncoder, hourEncoder, labelEncoder, classes, trainMean, colnames, trainMean = pickle.load(file)


file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/vars.obj",'rb')
MAX_POST_TEXT_LENGTH, MAX_TARGET_TITLE_LENGTH, MAX_TARGET_DESCRIPTION_LENGTH, MAX_TARGET_KEYWORDS_LENGTH, MAX_TARGET_PARAGRAPH_LENGTH = pickle.load(file)


### 1.) Models used at challenge
postTextBranch =  load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/postTextBranchReg.h5')
targetTitleBranch = load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/targetTitleBranchReg.h5')
descriptionBranch = load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/descriptionBranchReg.h5')
keywordsBranch = load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/keywordsBranchReg.h5')
paragraphBranch = load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/paragraphBranchReg.h5')
hourBranch = load_model('/home/philippe/PycharmProjects/deepLearning/clickbait/models/hourBranchReg.h5')

# load YAML and create model
yaml_file = open('/home/philippe/PycharmProjects/deepLearning/clickbait/models/modelReg.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_model = model_from_yaml(loaded_model_yaml)
final_model.load_weights("/home/philippe/PycharmProjects/deepLearning/clickbait/models/weightReg.h5")

# load YAML and create model
yaml_file = open('/home/philippe/PycharmProjects/deepLearning/clickbait/models/modelRegFull.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
final_model2 = model_from_yaml(loaded_model_yaml)
final_model2.load_weights("/home/philippe/PycharmProjects/deepLearning/clickbait/models/weightRegFull.h5")


folder = "/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-train-170331/"
instancesMap = {}
with open(folder +"instances.jsonl",'rb') as file:
    for line in file:
        instance = parseData(line.decode('utf-8'))
        instancesMap[instance._id] = instance

groundTruthMap = {}
with open(folder +"truth.jsonl",'rb') as file:
    for line in file:
        instance = parseGT(line.decode('utf-8'))
        groundTruthMap[instance._id] = instance


###Extract relevant parts and store in list...

testIDs = []
testPostText = []      # "text of the post with links removed (e.g., The 15-year-old has been detained for 21 months.)
testPostMedia = []     # path to a file in the media archive  (Single?)
testTitle = []         #"<title of target article>", (Single?)
testDescr = []         #"<description tag of target article>",
testKeywords = []      #"<keywords tag of target article>",
testParagraphs = []    # ["<text of the ith paragraph in the target article>"],
testCaptions   = []    #["<caption of the ith image in the target article>"]
testTime = []           #"<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",

for key in instancesMap:
    testIDs.append(str(instancesMap[key]._id))
    testPostText.append(str(instancesMap[key]._postText))
    testPostMedia.append(str(instancesMap[key]._postMedia))
    testTitle.append(str(instancesMap[key]._targetTitle))
    testDescr.append(str(instancesMap[key]._targetDescription))
    testKeywords.append(str(instancesMap[key]._targetKeywords))
    testParagraphs.append(str(instancesMap[key]._targetParagraphs))
    testCaptions.append(str(instancesMap[key]._targetCaptions))
    testTime.append(datetime.strptime(instancesMap[key]._postTimestamp,
                                  '%a %b %d %H:%M:%S +0000 %Y'))  # %a - Weekday; %b month name; %d day; %H Hour (24-hour clock): %M

#############################

##Predict using model model
def predictToFile(predictions, predictToFile):
    out_file = open(predictToFile, "w")
    for i in range(predictions.shape[0]):
        id = testIDs[i]
        predValue = float(predictions[i])
        predValue = max(0, predValue)
        predValue = min(1, predValue)

        my_dict = {
            'id': id,
            'clickbaitScore': predValue,
        }
        # print(placeName +" " +instance.text)
        json.dump(my_dict, out_file)
        out_file.write("\n")
    out_file.close()

def evaluatePrediction(predictions, goldLabel=groundTruthMap):
    y_true=[]
    y_pred=[]
    for i in range(predictions.shape[0]):
        id = testIDs[i]
        predValue = float(predictions[i])
        predValue = max(0, predValue)
        predValue = min(1, predValue)

        y_pred.append(float(predValue))
        y_true.append(float(groundTruthMap[id]._truthMean))
    mse =round(mean_squared_error(y_true, y_pred), ndigits=3)
    absv= round(median_absolute_error(y_true, y_pred), ndigits=3)
    print("mse=" +str(mse) +" abs=" +str(absv))
    print()

#1.) postTextModel
postTextSequence = postTextTokenizer.texts_to_sequences(testPostText)
postTextSequence = np.asarray(postTextSequence)  # Convert to ndArray
postTextSequence = pad_sequences(postTextSequence, maxlen=MAX_POST_TEXT_LENGTH)

predict = postTextBranch.predict(postTextSequence)
print("PostText")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )






#2.) targetTitle
titleSequence= targetTitleTokenizer.texts_to_sequences(testTitle)
titleSequence = np.asarray(titleSequence)  # Convert to ndArray
titleSequence = pad_sequences(titleSequence, maxlen=MAX_TARGET_TITLE_LENGTH)

predict = targetTitleBranch.predict(titleSequence)
print("targetTitle")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )



#3.) targetDescription
descriptionSequence= targetDescriptionTokenizer.texts_to_sequences(testTitle)
descriptionSequence = np.asarray(descriptionSequence)  # Convert to ndArray
descriptionSequence = pad_sequences(descriptionSequence, maxlen=MAX_TARGET_DESCRIPTION_LENGTH)

predict = descriptionBranch.predict(descriptionSequence)
print("targetDescription")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )


#4.) trainKeywords
keywordsSequence= targetKeywordsTokenizer.texts_to_sequences(testKeywords)
keywordsSequence = np.asarray(keywordsSequence)  # Convert to ndArray
keywordsSequence = pad_sequences(keywordsSequence, maxlen=MAX_TARGET_KEYWORDS_LENGTH)

predict = keywordsBranch.predict(keywordsSequence)
print("keywords")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )




#5.) trainParagraphs
paragraphSequence= targetParagraphTokenizer.texts_to_sequences(testParagraphs)
paragraphSequence = np.asarray(paragraphSequence)  # Convert to ndArray
paragraphSequence = pad_sequences(paragraphSequence, maxlen=MAX_TARGET_PARAGRAPH_LENGTH)

predict = paragraphBranch.predict(paragraphSequence) #(2459, 421)
print("trainParagraphs")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )


#7.) hour
testHour = list(map(lambda x: str(x.hour), testTime))
testHour = hourEncoder.transform(testHour)
testHour = np_utils.to_categorical(testHour)

predict = hourBranch.predict(testHour)
print("hour")
evaluatePrediction(predict)
#predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred.json")
#subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred.json"] )


##Final Model
predict = final_model.predict([postTextSequence, titleSequence, descriptionSequence, keywordsSequence, paragraphSequence, testHour])
evaluatePrediction(predict)

##Final Model2
predict = final_model2.predict([postTextSequence, titleSequence, descriptionSequence, keywordsSequence, paragraphSequence, testHour])
evaluatePrediction(predict)




#Final Model
predict = final_model.predict([postTextSequence, titleSequence, descriptionSequence, keywordsSequence, paragraphSequence, testHour])
predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred1.json")
subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred1.json"] )



#Final2 Model
predict = final_model2.predict([postTextSequence, titleSequence, descriptionSequence, keywordsSequence, paragraphSequence, testHour])
predictToFile(predictions=predict, predictToFile="/home/philippe/PycharmProjects/deepLearning/pred2.json")
subprocess.call(['/home/philippe/PycharmProjects/deepLearning/clickbait/eval.py', folder +"truth.jsonl", "/home/philippe/PycharmProjects/deepLearning/pred2.json"] )

