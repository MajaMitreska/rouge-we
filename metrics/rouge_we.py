
from collections import Counter
from sumeval.metrics.lang.base_lang import BaseLang
from sumeval.metrics.lang import get_lang

## Mathematical libraries
import numpy
import math

from metrics.word2vec import word2vec_model




class RougeCalculator():

    def __init__(self,
                 stopwords=True, stemming=False,
                 word_limit=-1, length_limit=-1, lang="en"):
        self.stemming = stemming
        self.stopwords = stopwords
        self.word_limit = word_limit
        self.length_limit = length_limit
        if isinstance(lang, str):
            self.lang = lang
            self._lang = get_lang(lang)
        elif isinstance(lang, BaseLang):
            self.lang = lang.lang
            self._lang = lang

    def tokenize(self, text_or_words, is_reference=False):
        """
        Tokenize a text under original Perl script manner.
        Parameters
        ----------
        text_or_words: str or str[]
            target text or tokenized words.
            If you use tokenized words, preprocessing is not applied.
            It allows you to calculate ROUGE under your customized tokens,
            but you have to pay attention to preprocessing.
        is_reference: bool
            for reference process or not
        See Also
        --------
        https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1820
        """
        words = text_or_words

        def split(text):
            _words = self._lang.tokenize(text)
            return _words

        if self.word_limit > 0:
            if isinstance(words, str):
                words = split(words)
            words = words[:self.word_limit]
            words = self._lang.join(words)
        elif self.length_limit > 0:
            text = words
            if isinstance(text, (list, tuple)):
                text = self._lang.join(words)
            words = text[:self.length_limit]

        if isinstance(words, str):
            words = self._lang.tokenize_with_preprocess(words)

        words = [w.lower().strip() for w in words if w.strip()]

        if self.stopwords:
            words = [w for w in words if not self._lang.is_stop_word(w)]

        if self.stemming and is_reference:
            # stemming is only adopted to reference
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1416

            # min_length ref
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L2629
            words = [self._lang.stemming(w, min_length=3) for w in words]
        return words

    def parse_to_be(self, text, is_reference=False):
        bes = self._lang.parse_to_be(text)

        def preprocess(be):
            be.head = be.head.lower().strip()
            be.modifier = be.modifier.lower().strip()
            if self.stemming and is_reference:
                be.head = self._lang.stemming(be.head, min_length=3)
                be.modifier = self._lang.stemming(be.modifier, min_length=3)

            return be

        bes = [preprocess(be) for be in bes]
        return bes

    # neka on_word2vec_diff raboti samo so summary ili reference
    # koga se samo so po eden zbor ili sporeduva zbor od summ so zbor od reference
    # ili koga se po dva zbora prateni, 2-gram od summ so 2-gram od ref
    # range from 0 to 1, 0 = least similar, 1 = the most similar
    # if the angle between two vectors is 90°, then the similarity would be 0.
    # For two vectors with an angle greater than 90°, then we also consider those to be 0

    def on_word2vec_diff(self, ngram1, ngram2):

        proceed = True

        word1 = str(ngram1)  # celata recenica
        word1_tokens = word1.split(" ") # ja deli  na tokeni
        word1_vectors = [ ]  # prazna niza vo koja ke se cuva vektorot za sekoj token
        for token in word1_tokens:
            # ako tokenot go nema vo word2vec modelot
            if not token in word2vec_model:
                word2vec_sim = 0
                proceed = False
                break
            # a, ako go ima go dodavame na nizara word1_vectors
            # word1_vectors.append(numpy.array(word2vec_model[token]))
            word1_vectors.append(word2vec_model[token])

        word2 = str(ngram2)
        word2_tokens = word2.split(" ")
        word2_vectors = [ ]
        for token in word2_tokens:
            if not token in word2vec_model:
                word2vec_sim = 0
                proceed = False
                break
            # word2_vectors.append(numpy.array(word2vec_model[token]))
            word2_vectors.append(word2vec_model[token])



        if proceed:
            # Se presmetuva slicnost
            word1_aggregate = word1_vectors[0]
            for i in range(1 ,len(word1_vectors)):
                # vo word1_aggregate gi mnozite site vektori od word1_vector nizata
                word1_aggregate *= word1_vectors[i]

            word2_aggregate = word2_vectors[0]
            for i in range(1, len(word2_vectors)):
                word2_aggregate *= word2_vectors[i]

            mag_word1vec = math.sqrt(numpy.dot(word1_aggregate, word1_aggregate))
            mag_word2vec = math.sqrt(numpy.dot(word2_aggregate, word2_aggregate))
            cosine_value = numpy.dot(word1_aggregate, word2_aggregate ) / (mag_word1vec * mag_word2vec)

            word2vec_sim = cosine_value

            # print("Similarity between " + word1 +" and "+word2+" is:" + str(cosine_value))

            return cosine_value

    def len_ngram(self, words, n):
        return max(len(words) - n + 1, 0)

    def ngram_iter(self, words, n):
        for i in range(self.len_ngram(words, n)):
            n_gram = words[i: i +n]
            yield tuple(n_gram)

    def count_ngrams(self, words, n):
        c = Counter(self.ngram_iter(words, n))
        return c



    def rouge_we_1(self, summary, references, alpha=0.5):
        return self.rouge_we(summary, references, 1, alpha)

    def rouge_we_2(self, summary, references, alpha=0.5):
        return self.rouge_we(summary, references, 2, alpha)

    def rouge_we(self, summary, references, n, alpha=0.5):
        """
        Calculate ROUGE-WE score.
        Parameters
        ----------
        summary: str
            summary text
        references: str
            concatenated references to evaluate summary
        n: int
            ROUGE kind. n=1, calculate when ROUGE-1
        alpha: float (0~1)
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))
        Returns
        -------
        f1: float
            f1 score
        """

        # _variable = it means that the varibale is for internal use, it is a local varibale


        # summary = str
        # references = str
        _summary = self.tokenize(summary)
        summary_ngrams = self.count_ngrams(_summary, n)
        _refs = self.tokenize(references)
        ref_ngrams = self.count_ngrams(_refs, n)

        recall_score = 0
        count_for_recall = 0
        score = 0

        for i in range(len(summary_ngrams)):
            score += self.on_word2vec_diff(summary_ngrams[i], ref_ngrams[i])

        count_for_recall = self.len_ngram(_refs, n)
        count_for_prec = self.len_ngram(_summary, n)
        f1 = self.calc_f1(score, count_for_recall, count_for_prec, alpha)
        return f1

    def calc_f1(self, score, count_for_recall, count_for_precision, alpha):
        def safe_div(x1, x2):
             return 0 if x2 == 0 else x1 / x2

        recall = safe_div(score, count_for_recall)
        precision = safe_div(score, count_for_precision)
        denom = (1.0 - alpha) * precision + alpha * recall

        return safe_div(precision * recall, denom)
