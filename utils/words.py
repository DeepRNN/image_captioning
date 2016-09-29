import os
import numpy as np
import cPickle as pickle

class WordTable():
    def __init__(self, dim_embed, max_sent_len, save_file):
        self.idx2word = []
        self.word2idx = {}
        self.word2vec = {}
        self.word_freq = []
        self.num_words = 0
        self.dim_embed = dim_embed
        self.max_sent_len = max_sent_len
        self.save_file = save_file

    def load_glove(self, glove_dir):
        self.word2vec = {}
        glove_file = os.path.join(glove_dir, 'glove.6B.'+str(self.dim_embed)+'d.txt')
        print("Loading Glove data from %s" %(glove_file))

        with open(glove_file) as f:
            for line in f:
                l = line.split()
                self.word2vec[l[0]] = [float(x) for x in l[1:]]

        print("Glove data loaded")

    def build(self, sentences):
        word_count = {}
        for sent in sentences:
            for w in sent.lower().split(' '):
                word_count[w] = word_count.get(w, 0) + 1
                if w not in self.word2vec:
                    self.word2vec[w] = np.random.uniform(0.0, 1.0, (self.dim_embed))

        sorted_word_count = sorted(list(word_count.items()), key=lambda x: x[1], reverse=True)        
        self.num_words = len(sorted_word_count)

        for idx in range(self.num_words):
            word, freq = sorted_word_count[idx]
            self.idx2word.append(word)
            self.word2idx[word] = idx
            self.word_freq.append(freq * 1.0)

        self.word_freq = np.array(self.word_freq)
        self.word_freq /= np.sum(self.word_freq)
        self.word_freq = np.log(self.word_freq)
        self.word_freq -= np.max(self.word_freq)

        self.filter_word2vec()

    def filter_word2vec(self):
        word2vec = {}
        for w in self.word2idx:
            word2vec[w] = self.word2vec[w] 
        self.word2vec = word2vec

    def symbolize_sent(self, sent):
        indices = np.zeros(self.max_sent_len).astype(np.int32)
        masks = np.zeros(self.max_sent_len)
        words = np.array([self.word2idx[w] for w in sent.lower().split(' ')])
        indices[:len(words)] = words
        masks[:len(words)] = 1.0
        return indices, masks

    def indices_to_sent(self, indices):
        words = [self.idx2word[i] for i in indices]
        if words[-1] != '.':
            words.append('.')
        punctuation = np.argmax(np.array(words) == '.') + 1
        words = words[:punctuation]
        res = ' '.join(words)
        res = res.replace(' .', '.')
        return res

    def save(self):
        pickle.dump([self.idx2word, self.word2idx, self.word2vec, self.word_freq, self.num_words], open(self.save_file, 'wb'))

    def load(self):
        self.idx2word, self.word2idx, self.word2vec, self.word_freq, self.num_words = pickle.load(open(self.save_file, 'rb'))

