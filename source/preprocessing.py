import nlp_util
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

## import dependecies
from sklearn.pipeline import Pipeline
from gensim.models import Phrases
from gensim.models.phrases import Phraser

# mport spacy, en_core_web_sm

# nlp=en_core_web_sm.load()
## Create corenlp object to process NLP
corenlp = nlp_util.NLP_Task()
# return POS grouped by unigrams, bigrams, trigrams using a dictionary

##
def preprocessing_txt(dataset):
    stop_words = set(stopwords.words('english'))
    corpus=[]
    for elm in range(0, len(dataset.index)):
        res=' '.join([i for i in dataset['Statement'][elm].lower().split() if i not in stop_words])
        res=re.sub("</?.*?>"," <> ",dataset['Statement'][elm])    # remove tags
        res=re.sub("(\\d|\\W)+"," ",dataset['Statement'][elm])        # remove special characte
        res=re.sub(r'@([A-Za-z0-9_]+)', "",dataset['Statement'][elm])  # remove twitter handler
        res=re.sub('(\r)+', "", dataset['Statement'][elm])            # remove newline character
        res=re.sub('[^\x00-\x7F]+', "", dataset['Statement'][elm])    # remove non-ascii characters
        res=''.join(x for x in dataset['Statement'][elm] if x not in set(string.punctuation))   ## remove punctuation
        corpus.append(res)
    return corpus


def sort_coo(coo_matrix):
    tuples=zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_top_words(feature_names, sorted_items, topn=3):
    sorted_items = sorted_items[:topn]
    score_vals,feature_vals,results = [],[],{}
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

##
def get_keywords_tfidf(dataset):
    keywords=[]
    cv=CountVectorizer(max_df=0.85, stop_words=None)
    tfidf_trans=TfidfTransformer(smooth_idf=True, use_idf=True)
    txt_corpus = preprocessing_txt(dataset)
    word_count_vector=cv.fit_transform(txt_corpus)
    tfidf_trans.fit(word_count_vector)
    feature_name = cv.get_feature_names()
    for i in range(0, len(txt_corpus)):
        tfidf_vector = tfidf_trans.transform(cv.transform([txt_corpus[i]]))
        sorted_vector=sort_coo(tfidf_vector.tocoo())
        res=extract_top_words(feature_name, sorted_vector,3)
        keywords.append(res)
    return keywords

##
def get_keywords(dataset):
	## 
	keywords=get_keywords_tfidf(dataset)
	sentences=[]
	for i in range(0, len(keywords)):
		sent=' '.join(list(keywords[i].keys()))
		sentences.append(sent)
	return sentences


## return a list of statements cleaned
def clean_text(sentences, remove_punctuation = True, lower_case = False, stop_words = False):
	processed_text = []
	stop=set(stopwords.words('english'))
	rm_punct=re.compile('[{}]'.format(re.escape(string.punctuation)))
	for sentence in sentences:
		text_to_clean = sentence
		if remove_punctuation:
			text_to_clean = rm_punct.sub(' ', text_to_clean)
		if lower_case:
			patt = re.sub('[^a-zA-Z]', ' ', text_to_clean)
			text_to_clean = ' '.join(str(patt).lower().split())
		if stop_words:
			res=' '.join([i for i in text_to_clean.lower().split() if i not in stop])
			text_to_clean = res
		processed_text.append(text_to_clean.strip())
	return processed_text

def extract_POS(statements):
	print('Extracting POS Tags')
	pos_tags = corenlp.POS_tagging(statements,return_word_tag_pairs=False)
	bigrams_pos = corenlp.POS_groupping(pos_tags, grams=2)
	trigrams_pos = corenlp.POS_groupping(pos_tags, grams=3)
	#For experimenting
	#print("Stringify")
	#pos_tags = [" ".join(x).replace('PRP$','PRP_DOLLAR') for x in pos_tags]
	#bigrams_pos = [" ".join(x).replace('PRP$','PRP_DOLLAR') for x in bigrams_pos]
	#trigrams_pos = [" ".join(x).replace('PRP$','PRP_DOLLAR') for x in trigrams_pos]
	print('Finished')
	return pos_tags,bigrams_pos,trigrams_pos

# return labels for multiclass classification
# label_values dictionary representing the different values for every label in labels
def create_labels(labels, label_values):
	n_labels = list()
	for label in labels:
		n_labels.append(label_values[label])
	return n_labels

# return number of word by sentence
def word_counts(statements):
	#getting tokens by sentences
	print('Extracting tokens by sentences')
	tbs = corenlp.TokensBySentence(statements) # tokens by sentence
	print('Counting tokens')
	wc = [len(x) for x in tbs]
	print('Finished')
	return wc

# return sentences vectors for pos unigrams
def pos_vectors(pos_tags,vector_dictionary=None, count=False, return_dictionary = False):
	if vector_dictionary == None:
		vector_dictionary = corenlp.UniquePosTags(pos_tags)
	# One hot version of POS tags
	occurrence_vector = np.zeros((len(pos_tags),len(vector_dictionary)))
	# Frequency vector
	frequency_vector = np.zeros((len(pos_tags),len(vector_dictionary)))
	#print('Processing POS tags and creating vectors')
	for index, pos_t in enumerate(pos_tags):
		for each_pos in pos_t:
			if each_pos in vector_dictionary:
				# get the index of the tags vector
				v_index = [i for i,x in enumerate(vector_dictionary) if x == each_pos][0]
				occurrence_vector[index][v_index] = 1
				frequency_vector[index][v_index] +=1
	#print('Finished')
	if count:
		if return_dictionary:
			return occurrence_vector, frequency_vector, vector_dictionary
		else:
			return occurrence_vector, frequency_vector
	else:
		if return_dictionary:
			return occurrence_vector, vector_dictionary
		else:
			return occurrence_vector

##
# def bigphrase_tfidf_feats(dataset):
#     corpus=preprocessing_txt(dataset)
#     lemmetized_sent=[]
#     for each_sent in nlp.pipe(corpus, batch_size=50, n_threads=-1):
#         if each_sent.is_parsed:
#             res=[tok.lemma_ for tok in each_sent if not tok.is_punct or tok.is_space or tok.is_stop or tok.like_num]
#             lemmetized_sent.append(res)
#         else:
#             lemmetized_sent.append(None)
#     bigram=Phraser(Phrases(lemmetized_sent))
#     bigram_lem=list(bigram[lemmetized_sent])
#     parsed=[]
#     for k in range(0, len(bigram_lem)):
#         joined=' '.join(bigram_lem[k])
#         parsed.append(joined)
#     return parsed, bigram_lem
		



