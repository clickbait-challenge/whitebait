import pickle
import numpy as np
import json
from representation import parseJsonLine, Place, extractPreprocessUrl
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import pandas
import time
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, InputLayer, Dense, Merge, Reshape, Conv1D, BatchNormalization, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
import keras.layers
import gzip



from keras.callbacks import EarlyStopping
#1.) Load relevant processing data

#file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/processors.obj",'rb')
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630//binary/processors.obj",'rb')
postTextTokenizer, targetTitleTokenizer, targetDescriptionTokenizer, targetKeywordsTokenizer, targetParagraphTokenizer, dayEncoder, hourEncoder, labelEncoder, classes, trainMean, colnames, trainMean = pickle.load(file)

#file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/data.obj",'rb')
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630//binary/data.obj",'rb')
trainPostText, trainTitle, trainDescr, trainKeywords, trainParagraphs, trainWeekday, trainHour, trainPostMedia= pickle.load(file)

#file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/binaryFull/vars.obj",'rb')
file = open("/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/clickbait/clickbait17-validation-170630//binary/vars.obj",'rb')
MAX_POST_TEXT_LENGTH, MAX_TARGET_TITLE_LENGTH, MAX_TARGET_DESCRIPTION_LENGTH, MAX_TARGET_KEYWORDS_LENGTH, MAX_TARGET_PARAGRAPH_LENGTH = pickle.load(file)



# create the model
batch_size = 32
verbosity=2

dropout = 0.3; recDropout = 0.1;
nb_epoch = 100
embeddings = 100
activationFunction='linear' #linear
#activationFunction='sigmoid' #linear

callbacks = [
    EarlyStopping(monitor='loss', min_delta=1e-4, patience=6, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, epsilon=0.0001, patience=2, cooldown=1, verbose=1)
]



#################Load glove-embeddings:

import os
embeddings_index = {}

#GLOVE_DIR="/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/embeddings/glove.6B"
GLOVE_DIR="/media/philippe/5f695998-f5a5-4389-a2d8-4cf3ffa1288a/data/embeddings/glove.twitter.27B"
#f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'))
f = gzip.open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt.gz'),'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    if coefs.shape[0] == 100: 
        embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

tmpMatrix = np.zeros((len(embeddings_index), 100)) #Initialize tmpMatrix for mean
counter = 0
for val in embeddings_index.values():
    tmpMatrix[counter] = val
    counter = counter +1
meanValue =np.mean(tmpMatrix, axis=0)
del(tmpMatrix)



###########1.) Embedding representation for postTextModel
word_index = postTextTokenizer.word_index #mapping from tokens to IDs
postTextEmbedding = np.zeros((len(postTextTokenizer.word_index) + 1, 100)) #Initialize Embedding matrix
unknownWords = 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        postTextEmbedding[i] = embedding_vector
    else:
#        print("unknown token " +word +"\t" +str(i))
        postTextEmbedding[i] = meanValue
        unknownWords = unknownWords +1
print("Vector unknown for " + str(unknownWords) +" words" + str(round(unknownWords / postTextEmbedding.shape[0] * 100)) + "%")
print("Embeddings matrix of shape %s" % postTextEmbedding.shape.__str__())


###########2.) Embedding representation for targetTitle
word_index = targetTitleTokenizer.word_index #mapping from tokens to IDs
targeTitleEmbedding = np.zeros((len(targetTitleTokenizer.word_index) + 1, 100)) #Initialize Embedding matrix
unknownWords = 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        targeTitleEmbedding[i] = embedding_vector
    else:
#        print("unknown token " +word +"\t" +str(i))
        targeTitleEmbedding[i] = meanValue
        unknownWords = unknownWords +1
print("Vector unknown for " + str(unknownWords) +" words" + str(round(unknownWords / targeTitleEmbedding.shape[0] * 100)) + "%")
print("Embeddings matrix of shape %s" % targeTitleEmbedding.shape.__str__())


###########3.) Embedding representation for targetDescription
word_index = targetDescriptionTokenizer.word_index #mapping from tokens to IDs
targetDescriptionEmbedding = np.zeros((len(targetDescriptionTokenizer.word_index) + 1, 100)) #Initialize Embedding matrix
unknownWords = 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        targetDescriptionEmbedding[i] = embedding_vector
    else:
#        print("unknown token " +word +"\t" +str(i))
        targetDescriptionEmbedding[i] = meanValue
        unknownWords = unknownWords +1
print("Vector unknown for " + str(unknownWords) +" words" + str(round(unknownWords / targetDescriptionEmbedding.shape[0] * 100)) + "%")
print("Embeddings matrix of shape %s" % targetDescriptionEmbedding.shape.__str__())

###########4.) Embedding representation for targetKeywordsTokenizer
word_index = targetKeywordsTokenizer.word_index #mapping from tokens to IDs
targetKeywordEmbedding = np.zeros((len(targetKeywordsTokenizer.word_index) + 1, 100)) #Initialize Embedding matrix
unknownWords = 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        targetKeywordEmbedding[i] = embedding_vector
    else:
#        print("unknown token " +word +"\t" +str(i))
        targetKeywordEmbedding[i] = meanValue
        unknownWords = unknownWords +1
print("Vector unknown for " + str(unknownWords) +" words" + str(round(unknownWords / targetKeywordEmbedding.shape[0] * 100)) + "%")
print("Embeddings matrix of shape %s" % targetKeywordEmbedding.shape.__str__())


###targetParagraphTokenizer
word_index = targetParagraphTokenizer.word_index #mapping from tokens to IDs
targetParagraphEmbedding = np.zeros((len(targetParagraphTokenizer.word_index) + 1, 100)) #Initialize Embedding matrix
unknownWords = 0

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        targetParagraphEmbedding[i] = embedding_vector
    else:
#        print("unknown token " +word +"\t" +str(i))
        targetParagraphEmbedding[i] = meanValue
        unknownWords = unknownWords +1
print("Vector unknown for " + str(unknownWords) +" words" + str(round(unknownWords / targetParagraphEmbedding.shape[0] * 100)) + "%")
print("Embeddings matrix of shape %s" % targetParagraphEmbedding.shape.__str__())
##################Load glove-embeddings over


############1.) postTextModel
postTextBranch = Sequential()
postTextBranch.add(Embedding(len(postTextTokenizer.word_index)+1,  #7632
                             output_dim = embeddings,
                             weights=[postTextEmbedding],
                             input_length=MAX_POST_TEXT_LENGTH)
                   )
postTextBranch.add(SpatialDropout1D(rate=dropout))
postTextBranch.add(BatchNormalization())
postTextBranch.add(Dropout(dropout))
postTextBranch.add(LSTM(units=30, recurrent_dropout=recDropout))
postTextBranch.add(BatchNormalization())
postTextBranch.add(Dropout(dropout, name="postText"))

postTextBranch.add(Dense(1, activation=activationFunction))
postTextBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
textHistory = postTextBranch.fit(trainPostText, trainMean,
                                 epochs=nb_epoch, batch_size=batch_size,
                                 verbose=verbosity, callbacks=callbacks
                                 )
print("textBranch finished after " +str(time.time() - start))
postTextBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/postTextBranchRegNew.h5')




############2.) targetTitle
targetTitleBranch = Sequential()
targetTitleBranch.add(Embedding(len(targetTitleTokenizer.word_index)+1,                   #7108
                                output_dim = embeddings,
                                weights=[targeTitleEmbedding],
                             input_length=MAX_TARGET_TITLE_LENGTH)
                      )
targetTitleBranch.add(SpatialDropout1D(rate=dropout))
targetTitleBranch.add(BatchNormalization())
targetTitleBranch.add(Dropout(dropout))
targetTitleBranch.add(LSTM(units=30, recurrent_dropout=recDropout))
targetTitleBranch.add(BatchNormalization())
targetTitleBranch.add(Dropout(dropout, name="targetTitle"))

targetTitleBranch.add(Dense(1, activation=activationFunction))
targetTitleBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
textHistory = targetTitleBranch.fit(trainTitle, trainMean,
                                    epochs=nb_epoch, batch_size=batch_size,
                                    verbose=verbosity, callbacks=callbacks
                                    )
print("targetTitleBranch finished after " +str(time.time() - start))
targetTitleBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/targetTitleBranchRegNew.h5')


############3.) targetDescription
descriptionBranch = Sequential()
descriptionBranch.add(Embedding(len(targetDescriptionTokenizer.word_index) + 1,  #9679
                                output_dim = embeddings,
                                weights=[targetDescriptionEmbedding],
                             input_length=MAX_TARGET_DESCRIPTION_LENGTH)
                      )
descriptionBranch.add(SpatialDropout1D(rate=dropout))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(dropout))
descriptionBranch.add(LSTM(units=30, recurrent_dropout=recDropout))
descriptionBranch.add(BatchNormalization())
descriptionBranch.add(Dropout(dropout, name="descriptionBranch"))

descriptionBranch.add(Dense(1, activation=activationFunction))
descriptionBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
textHistory = descriptionBranch.fit(trainDescr, trainMean,
                                    epochs=nb_epoch, batch_size=batch_size,
                                    verbose=verbosity, callbacks=callbacks
                                    )
print("descriptionBranch finished after " +str(time.time() - start))
descriptionBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/descriptionBranchRegNew.h5')


############4.) trainKeywords
keywordsBranch = Sequential()
keywordsBranch.add(Embedding(len(targetKeywordsTokenizer.word_index) + 1,  #5813
                                output_dim = embeddings,
                             weights=[targetKeywordEmbedding],
                             input_length=MAX_TARGET_KEYWORDS_LENGTH)
                   )
keywordsBranch.add(SpatialDropout1D(rate=dropout))
keywordsBranch.add(BatchNormalization())
keywordsBranch.add(Dropout(dropout))
keywordsBranch.add(LSTM(units=30, recurrent_dropout=recDropout))
keywordsBranch.add(BatchNormalization())
keywordsBranch.add(Dropout(dropout, name="keywordsBranch"))

keywordsBranch.add(Dense(1, activation=activationFunction))
keywordsBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
textHistory = keywordsBranch.fit(trainKeywords, trainMean,
                                 epochs=nb_epoch, batch_size=batch_size,
                                 verbose=verbosity, callbacks=callbacks
                                 )
print("keywordsBranch finished after " +str(time.time() - start))
keywordsBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/keywordsBranchRegNew.h5')

############5.) trainParagraphs
paragraphBranch = Sequential()
paragraphBranch.add(Embedding(len(targetParagraphTokenizer.word_index) + 1,  #46835
                                output_dim = embeddings,
                              weights=[targetParagraphEmbedding],
                             input_length=MAX_TARGET_PARAGRAPH_LENGTH)
                    )
paragraphBranch.add(SpatialDropout1D(rate=dropout))
paragraphBranch.add(BatchNormalization())
paragraphBranch.add(Dropout(dropout))
paragraphBranch.add(LSTM(units=30, recurrent_dropout=recDropout))
paragraphBranch.add(BatchNormalization())
paragraphBranch.add(Dropout(dropout, name="paragraphBranch"))

paragraphBranch.add(Dense(1, activation=activationFunction))
paragraphBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
textHistory = paragraphBranch.fit(trainParagraphs, trainMean,
                                  epochs=nb_epoch, batch_size=batch_size,
                                  verbose=verbosity, callbacks=callbacks
                                  )
print("paragraphBranch finished after " +str(time.time() - start))
paragraphBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/paragraphBranchRegNew.h5')


#7.) hour
import math
hourBranch = Sequential()
hourBranch.add(InputLayer(input_shape=(trainHour.shape[1],)))
hourBranch.add(Dense(int(math.log2(trainHour.shape[1])), activation='relu'))
hourBranch.add(BatchNormalization())
hourBranch.add(Dropout(dropout, name="hour"))


hourBranch.add(Dense(1, activation=activationFunction))
hourBranch.compile(loss='mean_squared_error', optimizer='rmsprop')
start = time.time()
utcHistory = hourBranch.fit(trainHour, trainMean,
                            epochs=nb_epoch, batch_size=batch_size,
                            verbose=verbosity, callbacks=callbacks
                            )
print("utcBranch finished after " +str(time.time() - start))
hourBranch.save('/home/philippe/PycharmProjects/deepLearning/clickbait/models/hourBranchRegNew.h5')



# 10.) Trainable merged model
from keras.models import Model
model1 = Model(inputs=postTextBranch.input, outputs=postTextBranch.get_layer('postText').output)
model2 = Model(inputs=targetTitleBranch.input, outputs=targetTitleBranch.get_layer('targetTitle').output)
model3 = Model(inputs=descriptionBranch.input, outputs=descriptionBranch.get_layer('descriptionBranch').output)
model4 = Model(inputs=keywordsBranch.input, outputs=keywordsBranch.get_layer('keywordsBranch').output)
model5 = Model(inputs=paragraphBranch.input, outputs=paragraphBranch.get_layer('paragraphBranch').output)
model7 = Model(inputs=hourBranch.input, outputs=hourBranch.get_layer('hour').output)



merged = Merge([model1, model2, model3, model4, model5, model7], mode='concat', name="merged")
final_model = Sequential()
final_model.add(merged)
final_model.add(Dropout(dropout))
final_model.add(Dense(1, activation=activationFunction))
final_model.compile(loss='mean_squared_error', optimizer='rmsprop')



start = time.time()
finalHistory = final_model.fit([trainPostText, trainTitle, trainDescr, trainKeywords, trainParagraphs, trainHour], trainMean,
                               epochs=nb_epoch, batch_size=batch_size, verbose=verbosity, callbacks=callbacks
                               )
end = time.time()
print("final_model-full finished after " +str(end - start))


model_yaml = final_model.to_yaml()
with open("/home/philippe/PycharmProjects/deepLearning/clickbait/models/modelRegFullNew.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
final_model.save_weights('/home/philippe/PycharmProjects/deepLearning/clickbait/models/weightRegFullNew.h5')



###Full Model
# 9.) Merged model
from keras.models import Model
model1 = Model(inputs=postTextBranch.input, outputs=postTextBranch.get_layer('postText').output)
model2 = Model(inputs=targetTitleBranch.input, outputs=targetTitleBranch.get_layer('targetTitle').output)
model3 = Model(inputs=descriptionBranch.input, outputs=descriptionBranch.get_layer('descriptionBranch').output)
model4 = Model(inputs=keywordsBranch.input, outputs=keywordsBranch.get_layer('keywordsBranch').output)
model5 = Model(inputs=paragraphBranch.input, outputs=paragraphBranch.get_layer('paragraphBranch').output)
model7 = Model(inputs=hourBranch.input, outputs=hourBranch.get_layer('hour').output)

for layer in model1.layers:
    layer.trainable = False

for layer in model2.layers:
    layer.trainable = False

for layer in model3.layers:
    layer.trainable = False

for layer in model4.layers:
    layer.trainable = False

for layer in model5.layers:
    layer.trainable = False

for layer in model7.layers:
    layer.trainable = False




merged = Merge([model1, model2, model3, model4, model5, model7], mode='concat', name="merged")
final_model = Sequential()
final_model.add(merged)
final_model.add(Dropout(dropout))
final_model.add(Dense(1, activation=activationFunction))
final_model.compile(loss='mean_squared_error', optimizer='rmsprop')



start = time.time()
finalHistory = final_model.fit([trainPostText, trainTitle, trainDescr, trainKeywords, trainParagraphs, trainHour], trainMean,
                               epochs=nb_epoch, batch_size=batch_size, verbose=verbosity, callbacks=callbacks
                               )
end = time.time()
print("final_model finished after " +str(end - start))


model_yaml = final_model.to_yaml()
with open("/home/philippe/PycharmProjects/deepLearning/clickbait/models/modelRegNew.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
final_model.save_weights('/home/philippe/PycharmProjects/deepLearning/clickbait/models/weightRegNew.h5')


