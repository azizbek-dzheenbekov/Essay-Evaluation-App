import re
import enchant
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams
from lexicalrichness import LexicalRichness
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FeaturePreprocess():

    def __init__(self, data, cnf):
        self.data = data
        self.cnf = cnf

    def capitalized(self, text):
        mistake_counter = 0
        sentences = sent_tokenize(text)

        for i in sentences:
            if (i[0].islower()):
                mistake_counter+=1
                return (mistake_counter/len(sentences))
      
    def count_words(self, text):
        words = text.split()
        return len(words)

    def count_sentences(self, text):
        sentences = sent_tokenize(text)
        return len(sentences)
    
    def text_to_paragraph(self, text):
        return list(filter(None, text.splitlines())) 
    
    def paragraph_count(self, text):
        paragraphs = list(filter(None, text.splitlines()))
        paragraphs_list = [par for par in paragraphs if len(par.split()) > 3]
        return len(paragraphs_list)

    def avg_sentence_count_per_paragraph(self, text):
        num_sentences = []
        for par in text:
            if (len(par.split()) > 3):
                num_sentences.append(len(re.split(r'[.!?]+', par))-1)   
        return sum(num_sentences)/len(num_sentences)
    
    def has_short_paragraphs(self, text):
        num_sentences = []
        for par in text:
            if (len(par.split()) > 3):
                num_sentences.append(len(re.split(r'[.!?]+', par))-1)

        if any(number < 3 for number in num_sentences):
            short_par = 1
        else: short_par = 0

        return short_par

    def remove_punctuation(self, text):
        punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
        no_punct=[words for words in text if words not in punctuation]
        words_wo_punct=''.join(no_punct)
        return words_wo_punct
    
    def check_en_us_spelling(self, text):
        en_dict = enchant.Dict("en_US")

        misspelled_count = 0
        for word in text.split():
            if(en_dict.check(word) == False or word == 'u'):
                misspelled_count += 1

        return misspelled_count
    
    def spelling_mistakes_percent(self, text):
        return self.check_en_us_spelling(text)/self.count_words(text)
    
    def count_unique_connectives(self, text):

        with open(self.cnf.files.connectives, mode='r') as file:
            connectives = eval(file.read())

        connectives_used = []

        for word in text.lower().split():
            if word in connectives:
                connectives_used.append(word)

        connectives_counter = Counter(connectives_used)
        num_connectives_filtered = len({x: count for x, count in connectives_counter.items() if count < 10}) 
        return num_connectives_filtered
    
    def percentage_of_unique(self, text):
        num_words = len(text.split())
        num_unique_words = len(set(text.split()))
        return num_unique_words/num_words
    
    def count_linking_phrases(self, text):

        with open(self.cnf.files.two_word_phrases, mode='r') as file:
            two_word_phrases = eval(file.read())
    
        with open(self.cnf.files.three_word_phrases, mode='r') as file:
            three_word_phrases = eval(file.read())

        no_punc_text = re.sub(r"[^\w\d'\s]+",' ', text)
        no_punc_text = no_punc_text.lower()

        bigrams = [' '.join(grams) for grams in nltk.ngrams(no_punc_text.split(), 2)]
        used_two_word_phrases = set()

        for gram in bigrams:
            if gram in two_word_phrases:
                used_two_word_phrases.add(gram)

        num_used_two_word_phrases = len(used_two_word_phrases)

        trigrams = [' '.join(grams) for grams in nltk.ngrams(no_punc_text.split(), 3)]
        used_three_word_phrases = set()

        for gram in trigrams:
            if gram in three_word_phrases:
                used_three_word_phrases.add(gram)

        num_used_three_word_phrases = len(used_three_word_phrases)

        num_used_phrases = num_used_two_word_phrases + num_used_three_word_phrases

        return num_used_phrases
    
    def word_count_without_stopwords(self, text):
        stopwords = nltk.corpus.stopwords.words('english')
        text_no_stopwords = " ".join([word for word in  text.split() if word not in stopwords])
        words = text_no_stopwords.split()
        return len(words)
    
    def measure_textual_lexical_diversity(self, text):
        lex = LexicalRichness(text)
        return lex.mtld()
    
    def average_punctuation(self, text):
        punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_{|}~'
        punct=[punct for punct in text if punct in punctuation]
        punct_percent = len(punct)/self.count_words(text)/self.count_sentences(text)
        return punct_percent
    
    def word_length_avg(self, text):
        words = text.split()
        average = sum(len(word) for word in words) / len(words)
        return average
    
    def repeats_bigram_percent(self, text):
        bigrams = ngrams(text.split(), 2)
        counts = dict()
        count_repeats = 0
        count_words = 0
        for bigram in bigrams:
            if bigram in counts:
                counts[bigram] += 1
                count_repeats+=1
            else:
                counts[bigram] = 1
            count_words+=1
        return count_repeats/count_words
    
    def nouns_count(self, text):
        tags = nltk.pos_tag(nltk.word_tokenize(text))
        nouns = [token[0] for token in tags if token[1] in ['NN', 'NNS', 'NNP', 'NNPS']]
        return len(nouns)/self.count_words(text)
    
    def adjectives_count(self, text):
        tags = nltk.pos_tag(nltk.word_tokenize(text))
        adjectives = [token[0] for token in tags if token[1] in ['JJ', 'JJR', 'JJS']]
        return len(adjectives)/self.count_words(text)
    
    def articles_count(self, text):
        articles = ['a', 'an', 'the']
        articles_all = " ".join([word for word in text.split() if word in articles])
        articles = articles_all.split()

        tags = nltk.pos_tag(nltk.word_tokenize(text))
        nouns = [token[0] for token in tags if token[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

        return len(articles)/len(nouns) 
    
    def modal_count(self, text):
        tags = nltk.pos_tag(nltk.word_tokenize(text))
        modal = [token[0] for token in tags if token[1] in ['MD']]
        return len(modal)/self.count_words(text)
    
    def count_phrasal_verbs(self, text):
        with open(self.cnf.files.phrasal_verbs, mode='r') as file:
            phrasal_verbs = eval(file.read())

        phrase_count = 0
        for phrase in phrasal_verbs:
            phrase_count += text.lower().count(phrase)
        return phrase_count
    
    def text_to_sentences(self, text):
        text.replace('\n\n', ' ')
        sentences = sent_tokenize(text)
        return sentences
    
    def count_verb_tense(self, text):    
        verb_tags = set()
        tokenized = sent_tokenize(text)
        for i in tokenized:
            words_list = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words_list)

            for pair in tagged:
                if pair[1][:2] == 'VB':
                    verb_tags.add(pair[1])

        return len(verb_tags)
    
    def cos_similarity_par_pairs(self, paragraphs):
    
        count_vectorizer = CountVectorizer()
        cos_similarities_list = []

        if len(paragraphs) > 1:
            for current_par, next_par in zip(paragraphs, paragraphs[1:]):
                vector_matrix = count_vectorizer.fit_transform([current_par, next_par])
                cosine_similarity_matrix = cosine_similarity(vector_matrix)
                cos_similarities_list.append(cosine_similarity_matrix[0][1])
        else: 
            cos_similarities_list.append(0.0)
        return sum(cos_similarities_list)/len(cos_similarities_list)
    
    def foreign_word_count(self, text):
        tags = nltk.pos_tag(nltk.word_tokenize(text))
        foreign_word = [token[0] for token in tags if token[1] in ['FW']]
        return len(foreign_word)
    
    def number_of_tags(self, text):
        tokenized = sent_tokenize(text)
        number_of_tags=0
        for i in tokenized:
            words_list = nltk.word_tokenize(i)
            tokens, tags = zip(*nltk.pos_tag(words_list))
            posBigrams = list(ngrams(tags, 2))
            for tag1, tag2 in posBigrams:
                number_of_tags+=1
        return number_of_tags
    
    def tags(self, text):
        tagTags = defaultdict(Counter)
        tokenized = sent_tokenize(text)
        for i in tokenized:
            words_list = nltk.word_tokenize(i)
            tokens, tags = zip(*nltk.pos_tag(words_list))
            posBigrams = list(ngrams(tags, 2))
            for tag1, tag2 in posBigrams:
                tagTags[tag1][tag2] += 1
        return tagTags
    
    def parse_pos_tags(self, text, pair1, pair2, all_tags):        
        if (text[pair1][pair2]):
            num = text[pair1][pair2]
        else:
            num = 0 
        return num/all_tags
    
    def assign_parsed_tags(self):

        with open(self.cnf.files.pos_list, mode='r') as file:
            pos_check_list = eval(file.read())

        for pair in pos_check_list:
            self.data[str(pair)]= self.data.apply(lambda x: self.parse_pos_tags(x['tags'], pair[0],pair[1], x['number_of_tags']), axis=1)
    
    def create_features(self):
        self.data['capitalized_mistakes'] = self.data['full_text'].apply(self.capitalized)
        self.data.fillna(0, inplace=True)
        self.data['word_count'] = self.data['full_text'].apply(self.count_words)
        self.data['sentence_count'] = self.data['full_text'].apply(self.count_sentences)
        self.data['paragraphs'] = self.data['full_text'].apply(self.text_to_paragraph)
        self.data['paragraph_count'] = self.data['full_text'].apply(self.paragraph_count)
        self.data['avg_sentence_count_per_paragraph'] = self.data['paragraphs'].apply(self.avg_sentence_count_per_paragraph)
        self.data['has_short_paragraphs'] = self.data['paragraphs'].apply(self.has_short_paragraphs)
        self.data['full_text_wo_punct'] = self.data['full_text'].apply(self.remove_punctuation)
        self.data['spelling_mistakes_percent'] = self.data['full_text_wo_punct'].apply(self.spelling_mistakes_percent)
        self.data.drop('full_text_wo_punct', axis=1, inplace=True)
        self.data['unique_linking_words_count'] = self.data['full_text'].apply(self.count_unique_connectives)
        self.data['percentage_of_unique_words'] = self.data['full_text'].apply(self.percentage_of_unique)
        self.data['linking_phrases_count'] = self.data['full_text'].apply(self.count_linking_phrases)
        self.data['word_count_no_stopwords'] = self.data['full_text'].apply(self.word_count_without_stopwords)
        self.data['mtld'] = self.data['full_text'].apply(self.measure_textual_lexical_diversity)
        self.data['average_punctuation'] = self.data['full_text'].apply(self.average_punctuation)
        self.data['word_length_avg'] = self.data['full_text'].apply(self.word_length_avg)
        self.data['repeats_bigram_percent'] = self.data['full_text'].apply(self.repeats_bigram_percent)
        self.data['nouns_count'] = self.data['full_text'].apply(self.nouns_count)
        self.data['adjectives_count'] = self.data['full_text'].apply(self.adjectives_count)
        self.data['articles_count'] = self.data['full_text'].apply(self.articles_count)
        self.data['modal_count'] = self.data['full_text'].apply(self.modal_count)
        self.data["phrasal_verbs_count"] = self.data["full_text"].apply(self.count_phrasal_verbs)
        self.data['sentences'] = self.data['full_text'].apply(self.text_to_sentences)
        self.data["verb_tense_count_percent"] = self.data["full_text"].apply(self.count_verb_tense) / self.data['word_count']
        self.data['avg_cos_similarity'] = self.data['paragraphs'].apply(self.cos_similarity_par_pairs)
        self.data['foreign_word_count'] = self.data['full_text'].apply(self.foreign_word_count) / self.data['word_count']
        self.data['number_of_tags'] = self.data['full_text'].apply(self.number_of_tags)
        self.data['tags'] = self.data['full_text'].apply(self.tags)
        self.assign_parsed_tags()
        self.data.drop(columns=['paragraphs', 'word_count','sentences', 'tags'], inplace=True)