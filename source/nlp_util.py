from pycorenlp import StanfordCoreNLP
import numpy as np
import string

class NLP_Task:
	"""
		Initialize StanfordCoreNLP
	"""
	def __init__(self):
			self.core_nlp = StanfordCoreNLP('http://localhost:9000')
			print("NLP_Task ready to use.")

	"""
		return POS tags ngram wise
	"""
	def POS_tagging(self, statements, return_word_tag_pairs = False):
		POS_tags = list()
		for statement in statements:
			statement_tags = list()
			annotations = self.core_nlp.annotate(statement, properties={
			  'annotators': 'tokenize,pos',
			  'outputFormat': 'json'
			  })
			for output in annotations['sentences']:
				statement_tags.append('<s>')
				previous = ''
				for token in output['tokens']:
					if return_word_tag_pairs:
						statement_tags.append(token['word']+'/'+token['pos'])
					else:
						statement_tags.append(token['pos'])

			POS_tags.append(statement_tags)
		return POS_tags

	

	"""
		APPROACH TO REMOVE TWO OR MORE NP appearing together
		Also, two or more CD appearing together: Usually 22 Millions is tagged as CD CD and for the sake of analysis we want to group the as CD
	"""
	def RemoveConsecutiveTags(self,list_to_remove, postags,ignore_punctuation=False):
		withoutConsecutiveTags = list()
		for each_tag in postags:
			removed = list()
			previous = ''
			for tt in each_tag:
				if tt != previous:
					if not ignore_punctuation: # ignore punctuation, add it as previous POS but do not add it to the final list
						removed.append(tt)
					elif tt not in string.punctuation:
						removed.append(tt)
					previous = tt
				elif tt not in list_to_remove:
					removed.append(tt)
					previous = tt
			withoutConsecutiveTags.append(removed)
		return withoutConsecutiveTags

	"""
		return POS grouped by number of grams
	"""
	def POS_groupping(self, sentences_pos,grams=1):
		result = list()
		for sentence_tags in sentences_pos:
			tag_group = list()
			for index, each_tag in enumerate(sentence_tags):
				if index < len(sentence_tags)-grams and len(sentence_tags)>=grams:
					format_str = str()
					for i in range(0,grams):
						format_str += sentence_tags[index+i]
						if i<grams-1:
							format_str += '_'
					tag_group.append(format_str)
			result.append(tag_group)
		return result

	"""
		return the list of tokens per statement
	"""
	def TokensBySentence(self, sentences):
		tbs =  list()
		for sentence in sentences:
			token_list = list()
			output = self.core_nlp.annotate(sentence, properties={
			  'annotators': 'tokenize',
			  'outputFormat': 'json'
			  })
			for t in output['tokens']:
				token_list.append(t['word'])
			tbs.append(token_list)
		return tbs

	"""
		return the unique values for POS grouped by any
	"""
	def UniquePosTags(self, postags, return_counts = False):
		pos_list =list()
		counts = dict()
		for pos in postags:
			for p in pos:
				if p not in pos_list:
					pos_list.append(p)
				if p not in counts.keys():
					counts[p] = 1
				else:
					counts[p] += 1
		if return_counts:
			return pos_list, counts
		else:
			return pos_list