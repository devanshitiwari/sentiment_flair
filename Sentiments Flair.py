from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.data import Sentence
from flair.trainers import ModelTrainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from torch.optim.adam import Adam
import re, nltk, spacy

from flair.embeddings import TransformerDocumentEmbeddings

import pandas as pd
import numpy as np


#MODEL TRAINING
data = pd.read_csv('sentiment_regression_input.csv')

# It is a good idea to check and make sure the data is loaded as expected.

print(data.head(5))


# Pandas ".iloc" expects row_indexer, column_indexer
X = data['text']
# Now let's tell the dataframe which column we want for the target/labels.
y = data['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

output = pd.DataFrame()
output['text'] = X_train
output['sentiment'] = y_train
output.to_csv('train.csv', index=False)

output = pd.DataFrame()
output['text'] = X_val
output['sentiment'] = y_val
output.to_csv('dev.csv', index=False)

output = pd.DataFrame()
output['text'] = X_test
output['sentiment'] = y_test
output.to_csv('test.csv', index=False)


from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = './data_folder'

# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label_topic"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter=',',    # tab-separated files
)



# print the number of Sentences in the train split
print(len(corpus.train))

# print the number of Sentences in the test split
print(len(corpus.test))

# print the number of Sentences in the dev split
print(len(corpus.dev))



# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('distilbert-base-uncased', fine_tune=True)

# 4. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 5. initialize the text classifier trainer with Adam optimizer
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

# 6. start the training
trainer.train('./model_result',
              learning_rate=3e-5, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=5, # terminate after 5 epochs
              )

def clean(raw):
    """ Remove hyperlinks and markup """
    result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
    result = re.sub('&gt;', "", result)
    result = re.sub('&#x27;', "'", result)
    result = re.sub('&quot;', '"', result)
    result = re.sub('&#x2F;', ' ', result)
    result = re.sub('<p>', ' ', result)
    result = re.sub('</i>', '', result)
    result = re.sub('&#62;', '', result)
    result = re.sub('<i>', ' ', result)
    result = re.sub("\n", '', result)
    return result


classifier = TextClassifier.load('./model_result/final-model.pt')



df = pd.read_csv('output_sentiment_600.csv')
#df = df.head(10)

df['Text'] = df['Text'].fillna('').apply(str)




d = []

for i, row in df.iterrows():
    document =  row['Text']
    document = clean(document)
    sentence = Sentence(document)
    classifier.predict(sentence)
    print(document+"\n\n")
    print(sentence.labels)
    print("\n\n")
    d.append(
        {
            'Sentiment': sentence.labels
        }
    )

df['flair_sentiment'] = d
output = pd.DataFrame(df)
output.to_csv('output_sentiment_flair_600.csv', index=False)
