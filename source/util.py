import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
"""
	Load tsv files.
	return: train, validation and testing files
"""
def load_files(path='..\\dataset\\{0}', file_names=['train.tsv','valid.tsv','test.tsv']):
	# define columns names
	column_names = ['Id', 'Label','Statement','Subject','Speaker','Speaker Job','State Info','Party','BT','FC','HT','MT','PF','Context']

	# load each file
	train_file = pd.read_csv(path.format(file_names[0]), sep='\t', header=None, encoding='utf-8')
	validation_file = pd.read_csv(path.format(file_names[1]), sep='\t',header=None, encoding='utf-8')
	testing_file = pd.read_csv(path.format(file_names[2]), sep='\t',header=None, encoding='utf-8')

	# set columns names
	train_file.columns = column_names
	validation_file.columns = column_names
	testing_file.columns = column_names

	return train_file, validation_file, testing_file

def load_file(path, file_name):
	# define columns names
	#column_names = ['Id', 'Label','Statement','Subject','Speaker','Speaker Job','State Info','Party','BT','FC','HT','MT','PF','Context']

	# load each file
	uploading_file = pd.read_csv(path.format(file_name), encoding='utf-8')
	#validation_file = pd.read_csv(path.format(file_names[1]), sep='\t',header=None, encoding='utf-8')
	#testing_file = pd.read_csv(path.format(file_names[2]), sep='\t',header=None, encoding='utf-8')

	# set columns names
	#train_file.columns = column_names
	#validation_file.columns = column_names
	#testing_file.columns = column_names

	return uploading_file


"""
	Convert tsv file columns into a dictionary
	return: dictionary with specified columns
"""
def tsv_to_dict(tsv_file,columns=None):
	# columns to return 
	if columns==None:
		columns = tsv_file.columns

	# dictionary to return
	data_dict = dict()

	for each_column in columns:
		d_key = each_column.lower().strip().replace(' ','-')
		data_dict[d_key] = tsv_file[[each_column]].values[:,0] 

	return data_dict

"""
	return the most frequent element in a sequence
"""

# def mode(seq):
#   	if len(seq) == 0:
# 		  #
# 	  	  return 1.
#   	else:
# 		  cnt = {}
# 	for item in seq:
# 	  	if item in cnt:
# 			cnt[item] += 1
# 	  	else:
# 			cnt[item] = 1
# 		maxItem = seq[0]
# 	for item,c in cnt.items():
# 	  	if c > cnt[maxItem]:
# 			maxItem = item
# 	return maxItem


#use_built_in_vectors => if True use CountVectorizer, otherwise it uses the version of pos_vectors in preprocessing
def GetFeaturesFromPOS(training_data, validation_data, testing_data, user_defined_vocabulary=None):

	user_defined_vocabulary = [x.lower().replace('$','dollar') for x in user_defined_vocabulary]

	# making string of the data
	training_str = [" ".join(x) for x in training_data]
	validation_str = [" ".join(x) for x in validation_data]
	testing_str = [" ".join(x) for x in testing_data]

	#replace $ by dollar
	training_str = [x.replace('$', 'dollar').replace('<s>','sos') for x in training_str]
	validation_str = [x.replace('$', 'dollar').replace('<s>','sos') for x in validation_str]
	testing_str = [x.replace('$', 'dollar').replace('<s>','sos') for x in testing_str]

	# features using binary iformation
	oneHotVectorizer = CountVectorizer(vocabulary=user_defined_vocabulary,binary=True)
	tr_onehot = oneHotVectorizer.fit_transform(training_str).toarray()
	val_onehot = oneHotVectorizer.transform(validation_str).toarray()
	te_onehot = oneHotVectorizer.transform(testing_str).toarray()
	print(oneHotVectorizer.vocabulary_)

	# features using no-binary information (counting)
	countVectorizer = CountVectorizer(vocabulary=user_defined_vocabulary,binary=True)
	tr_count = countVectorizer.fit_transform(training_str).toarray()
	val_count = countVectorizer.transform(validation_str).toarray()
	te_count = countVectorizer.transform(testing_str).toarray()

	# features using tf-idf vectors

	tfIdfVectorizer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
	tr_tfidf = tfIdfVectorizer.fit_transform(tr_count)
	val_tfidf = tfIdfVectorizer.transform(val_count)
	te_tfidf =  tfIdfVectorizer.transform(te_count)

	return tr_onehot, tr_count, tr_tfidf, val_onehot, val_count, val_tfidf, te_onehot, te_count, te_tfidf