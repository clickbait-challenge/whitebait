import json
import numpy as np
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import pandas
from sklearn.preprocessing import LabelEncoder


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


folder = "/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630/"

instancesMap = {}
with open(folder +"instances.jsonl",'rb') as file:
    for line in file:
        instance = parseData(line.decode('utf-8'))
        instancesMap[instance._id] = instance


with open(folder +"truth.jsonl",'rb') as file:
    for line in file:
        gtInstance = parseGT(line.decode('utf-8'))
        instancesMap[gtInstance._id]._groundTruth = gtInstance

from collections import Counter
lorum = list(map(lambda x: len(x._postMedia), list(instancesMap.values())))
Counter(lorum)



folder = "/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-train-170331/"
with open(folder +"instances.jsonl",'rb') as file:
    for line in file:
        instance = parseData(line.decode('utf-8'))
        instancesMap[instance._id] = instance


with open(folder +"truth.jsonl",'rb') as file:
    for line in file:
        gtInstance = parseGT(line.decode('utf-8'))
        instancesMap[gtInstance._id]._groundTruth = gtInstance


###Extract relevant parts and store in list...
trainLabels = []  # list of label ids
trainMedian = []  #Average scores
trainMean = []

trainTime = []           #"<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",
trainPostText = []      # "text of the post with links removed (e.g., The 15-year-old has been detained for 21 months.)
trainPostMedia = []     # path to a file in the media archive  (Single?)
trainTitle = []         #"<title of target article>", (Single?)
trainDescr = []         #"<description tag of target article>",
trainKeywords = []      #"<keywords tag of target article>",
trainParagraphs = []    # ["<text of the ith paragraph in the target article>"],
trainCaptions   = []    #["<caption of the ith image in the target article>"]
for key in instancesMap:
    trainLabels.append(instancesMap[key]._groundTruth)
    trainMedian.append(instancesMap[key]._groundTruth._truthMedian)
    trainMean.append(instancesMap[key]._groundTruth._truthMean)

    trainTime.append(datetime.strptime(instancesMap[key]._postTimestamp, '%a %b %d %H:%M:%S +0000 %Y'))  # %a - Weekday; %b month name; %d day; %H Hour (24-hour clock): %M
    trainPostText.append(str(instancesMap[key]._postText))
    trainPostMedia.append(str(instancesMap[key]._postMedia))
    trainTitle.append(str(instancesMap[key]._targetTitle))
    trainDescr.append(str(instancesMap[key]._targetDescription))
    trainKeywords.append(str(instancesMap[key]._targetKeywords))
    trainParagraphs.append(str(instancesMap[key]._targetParagraphs))
    trainCaptions.append(str(instancesMap[key]._targetCaptions))

#trainNumbers = list(map(lambda x: float(x._truthMean), trainLabels)) # For regression
trainClasses = list(map(lambda x: float(x._truthMedian), trainLabels)) #For classification


#1.) Binarize > 4 classes into one hot encoding ({0.6666667, 1.0, 0.0, 0.33333334})
labelEncoder = LabelEncoder()
labelEncoder.fit(trainClasses)
classes = labelEncoder.transform(trainClasses)

#Map class int representation to
colnames = [None]*len(set(classes))
for i in range(len(classes)):
    colnames[classes[i]] =  trainLabels[i]._truthMedian


#2.) Tokenize texts
import string
def my_filter():
    f = """’!"$%&'()*+,-./:;<=>?@[\]^_`{|}~""" # Basically string.punctuation without Hashtags
    f += '\t\n\r…“”‘'
    return f

#1.) postText
MAX_POST_TEXT_LENGTH=13
postTextTokenizer = Tokenizer(filters=my_filter()) #Keep only top-N words
postTextTokenizer.fit_on_texts(trainPostText)
trainPostText = postTextTokenizer.texts_to_sequences(trainPostText)
print("Median = " +str(sorted(list(map(lambda x: len(x), trainPostText)))[int(len(trainPostText)/2)]) )
trainPostText = np.asarray(trainPostText)
trainPostText = pad_sequences(trainPostText, maxlen=MAX_POST_TEXT_LENGTH)


#2.) targetTitle
MAX_TARGET_TITLE_LENGTH=11
targetTitleTokenizer = Tokenizer(filters=my_filter()) #Keep only top-N words
targetTitleTokenizer.fit_on_texts(trainTitle)
trainTitle = targetTitleTokenizer.texts_to_sequences(trainTitle)
print("Median = " +str(sorted(list(map(lambda x: len(x), trainTitle)))[int(len(trainTitle)/2)]) )
trainTitle = np.asarray(trainTitle)
trainTitle = pad_sequences(trainTitle, maxlen=MAX_TARGET_TITLE_LENGTH)

#3.) targetDescription
MAX_TARGET_DESCRIPTION_LENGTH=23
targetDescriptionTokenizer = Tokenizer(filters=my_filter()) #Keep only top-N words
targetDescriptionTokenizer.fit_on_texts(trainDescr)
trainDescr = targetDescriptionTokenizer.texts_to_sequences(trainDescr)
print("Median = " +str(sorted(list(map(lambda x: len(x), trainDescr)))[int(len(trainDescr)/2)]) )
trainDescr = np.asarray(trainDescr)
trainDescr = pad_sequences(trainDescr, maxlen=MAX_TARGET_DESCRIPTION_LENGTH)

#4.) targetKeywordsTokenizer
MAX_TARGET_KEYWORDS_LENGTH=7
targetKeywordsTokenizer = Tokenizer(filters=my_filter()) #Keep only top-N words
targetKeywordsTokenizer.fit_on_texts(trainKeywords)
trainKeywords = targetKeywordsTokenizer.texts_to_sequences(trainKeywords)
print("Median = " +str(sorted(list(map(lambda x: len(x), trainKeywords)))[int(len(trainKeywords)/2)]) )
trainKeywords = np.asarray(trainKeywords)
trainKeywords = pad_sequences(trainKeywords, maxlen=MAX_TARGET_KEYWORDS_LENGTH)

#5.) targetParagraphs
MAX_TARGET_PARAGRAPH_LENGTH=450
targetParagraphTokenizer = Tokenizer(filters=my_filter()) #Keep only top-N words
targetParagraphTokenizer.fit_on_texts(trainParagraphs)
trainParagraphs = targetParagraphTokenizer.texts_to_sequences(trainParagraphs)
print("Median = " +str(sorted(list(map(lambda x: len(x), trainParagraphs)))[int(len(trainParagraphs)/2)]) )
trainParagraphs = np.asarray(trainParagraphs)
trainParagraphs = pad_sequences(trainParagraphs, maxlen=MAX_TARGET_PARAGRAPH_LENGTH)

##6.) targetCaptions

#7.) trainTime
from collections import Counter
Counter(list(map(lambda x: (x.year, x.month, x.day), trainTime)))
#Counter({(2015, 6, 12): 483, (2015, 6, 11): 448, (2015, 6, 9): 338, (2015, 6, 10): 312, (2015, 6, 13): 309, (2015, 6, 14): 280, (2015, 6, 8): 274, (2015, 6, 7): 15})
#Counter(list(map(lambda x: (x.hour), trainTime)))
trainHour = list(map(lambda x: str(x.hour), trainTime))
trainWeekday = list(map(lambda x: str(x.weekday()), trainTime))

dayEncoder = LabelEncoder()
dayEncoder.fit(trainWeekday)
testVar = dayEncoder.transform(trainWeekday)

categorial = np.zeros((len(testVar), len(dayEncoder.classes_)), dtype="int8")
for i in range(len(testVar)):
    categorial[i, testVar[i]] = 1
trainWeekday = categorial




hourEncoder = LabelEncoder()
hourEncoder.fit(trainHour)
testVar = hourEncoder.transform(trainHour)

categorial = np.zeros((len(testVar), len(hourEncoder.classes_)), dtype="int8")
for i in range(len(testVar)):
    categorial[i, testVar[i]] = 1
trainHour = categorial



import pickle
#1.) Save relevant processing data
filehandler = open(b"/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/processors.obj","wb")
#filehandler = open(b"/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630/binary/processors.obj","wb")

pickle.dump((postTextTokenizer, targetTitleTokenizer, targetDescriptionTokenizer, targetKeywordsTokenizer, targetParagraphTokenizer, dayEncoder, hourEncoder, labelEncoder, classes, trainMedian, colnames, trainMean), filehandler)
filehandler.close()

#1.b) Save variables
filehandler = open(b"/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/vars.obj","wb")
#filehandler = open(b"/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630/binary/vars.obj","wb")
pickle.dump((MAX_POST_TEXT_LENGTH, MAX_TARGET_TITLE_LENGTH, MAX_TARGET_DESCRIPTION_LENGTH, MAX_TARGET_KEYWORDS_LENGTH, MAX_TARGET_PARAGRAPH_LENGTH), filehandler)
filehandler.close()

#2.) Save converted training data
