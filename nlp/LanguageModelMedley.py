import math, collections

class LanguageModelMedley:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.starts_with = collections.defaultdict(lambda: 0)
    self.ends_with = collections.defaultdict(lambda: 0)
    self.train(corpus)
    self.all_words = sum(self.unigramCounts.values())
    self.vocabulary_size = len(self.unigramCounts)
    self.bigram_total = len(self.bigramCounts)
    self.unigram_v_n = sum(self.unigramCounts.values()) + len(self.unigramCounts)

  def train(self, corpus):
    for sentence in corpus.corpus:
      previous_word = None
      i = 0
      for datum in sentence.data:
      # for i in range(1,len(sentence)):
        current_word = datum.word
        self.unigramCounts[current_word] += 1
        bigram = (previous_word,current_word)
        self.bigramCounts[bigram] += 1
        if self.bigramCounts[bigram] == 1:
          self.starts_with[bigram[0]] += 1 
          self.ends_with[bigram[1]] += 1
        i += 1

  def stupidBackOff(self, sentence):
    # unsmoothed bigram model combined with backoff to an add-one smoothed unigram model
    score = 0.0
    previous_token = None
    for token in sentence:
      key = (previous_token, token)
      if self.bigramCounts[key] > 0: # without smoothing
        # probabilty of c(Wn-i, Wn) / c(Wn-i)
        # c(Wn-i, Wn)
        score += math.log(self.bigramCounts[key])
        # c(Wn-1)
        if self.unigramCounts[previous_token]:
          prev_score = 0.0
          prev_score += math.log(self.unigramCounts[previous_token])
          score -= prev_score
        else:
          score = 0.0003
      else: # backoff
        score += math.log(self.unigramCounts[token] + 1) # smoothing
        score -= math.log(self.all_words + self.unigramCounts[token])
      previous_token = token
    return score

  def kneserNey(self, sentence):
    score = 0.0
    for i in xrange(1, len(sentence)):
      bigram = tuple(sentence[i-1:i+1])
      discount = 0.75
      p_wn = self.unigramCounts[bigram[0]] + 0.003
      prob = max(self.bigramCounts[bigram] - discount, 0) / p_wn
      if self.unigramCounts[bigram[0]] > 0:
        lambada = (discount / p_wn) * self.starts_with[bigram[0]]  
      else:
        lambada = discount * 0.003
      p_cont = float(self.ends_with[bigram[1]] + 0.01) / float(self.bigram_total)
      score += math.log(prob + (lambada * p_cont))
    return score

  def laplaceBigram(self, sentence):
    # PLaplace(wi|wi-1) = (1+c(wi-1, wi)) / V+c(wi-1)
    score = 0.0
    previous_token = None
    for token in sentence:
      bigram = (previous_token, token)
      print bigram
      score += math.log(self.bigramCounts[bigram] + 1) # 1 + c(wi) <-- smoothing
      score -= math.log(self.all_words + self.unigramCounts[previous_token])
      previous_token = token
    return score

  def laplaceUnigram(self, sentence):
    # PLaplace(wi) = (1 + c(wi)) / V + N
    score = 0.0 
    for token in sentence:
      score += math.log(self.unigramCounts[token] + 1) # 1 + c(wi) <-- smoothing
      score -= math.log(self.unigram_v_n)
    return score

  def unigram(self, sentence):
    score = 0.0 
    for token in sentence:
      count = self.unigramCounts[token]
      if count > 0:
        score += math.log(count)
        score -= math.log(self.all_words)
      else:
        score =  float('-inf') # not smoothed
    return score

  def uniform(self, sentence):
    score = 0.0
    probability = math.log(1.0/self.vocabulary_size)
    for token in sentence: # iterate over words in the sentence
      score += probability
    return score