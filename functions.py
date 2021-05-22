from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from nltk.util import ngrams 
import re
import random

def process_text(text):
  alphabet_list=list(set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя \t\n'))
  text_new = text.lower()
  list_text  = list(text_new)
  list_text_new = [c for c in list_text if c in alphabet_list]
  text_new = "".join(list_text_new)
  text_new = text_new.replace("\n", ' ')
  text_new = text_new.replace("\t", ' ')
  text_new = re.sub("\s+", " ",text_new)
  return text_new


def get_letters_freqs(text_new):
  text_list = list(text_new)
  cnts = Counter(text_list)
  
  total =  sum(cnts.values())

  cnts_freqs = {k: v/total for k, v in cnts.items()}
  return cnts, cnts_freqs


def plot_letters_freqs(cnts: dict):
  keys = sorted(cnts.keys(), key=lambda letter: cnts[letter], reverse=True)
  values = [cnts[key] for key in keys]
  plt.bar(keys, values, width = 2, color='g')


def get_diff_counts(alphabet_dict, text_dict) -> dict:
  cnts = alphabet_dict
  alphabet_dict_keys = sorted(cnts.keys(), key=lambda letter: cnts[letter], reverse=True) 
  cnts = text_dict
  text_dict_keys = sorted(cnts.keys(), key=lambda letter: cnts[letter], reverse=True) 
  diff_dict = [abs(alphabet_dict_keys.index(al) - text_dict_keys.index(al)) for al in text_dict_keys]
  return diff_dict, text_dict_keys, alphabet_dict_keys


  def l1_score(letter_freqs, decoded_text):
    bigrams_freqs = get_ngrams_freqs_in_words(decoded_text,  2)
    #print(bigrams_freqs)
    score = 0
    for key in letter_freqs.keys():
      #print(key)
      if key in bigrams_freqs.keys():
        score += abs(bigrams_freqs[key] - letter_freqs[key])
    return score    

def get_log_score(freqs_large, decoded_text, n_gram):
  freqs_decoded = get_ngrams_quantities_in_words(decoded_text,  n_gram)
  log_score = 0
  for item in freqs_decoded:
    if (item not in freqs_large) :
      continue
    #print(item, freqs_decoded[item], freqs_large[item] )
    #if (item in freqs_large) :
    log_score += freqs_decoded[item] * np.log(freqs_large[item])
  return log_score 
    
def get_log_score_shifr(freqs_large, coded_text, shifr_str, alphabet_str, n_gram):
  decoded_text = decode_text(coded_text, shifr_str, alphabet_str)
  #print(decoded_text)
  #freqs_large = get_ngrams_freqs(large_text,  n_gram)
  freqs_decoded = get_ngrams_quantities(decoded_text,  n_gram)
  log_score = 0
  for item in freqs_decoded:
    if (item not in freqs_large) :
      #log_score += freqs_decoded[item] * np.log(0.8)
      #print(item)
      continue
      #return (0)
    #print(item, freqs_decoded[item], freqs_large[item] )
    #if (item in freqs_large) :
    log_score += freqs_decoded[item] * np.log(freqs_large[item])
  return log_score      
    

def get_ngrams_quantities(text_new,  n_gram):
  ngrams_text = ngrams(list(text_new), n_gram)
  cnts = Counter(ngrams_text)
  #total =  sum(cnts.values())
  #cnts = {k: v/total for k, v in cnts.items()}
  #print(cnts)
  return cnts  

def get_ngrams_freqs(text_new,  n_gram):
  ngrams_text = ngrams(list(text_new), n_gram)
  cnts = Counter(ngrams_text)
  total =  sum(cnts.values())
  cnts = {k: v/total for k, v in cnts.items()}
  #print(cnts)
  return cnts  


def get_ngrams_quantities_in_words(text_new,  n_gram):
  words = text_new.strip().split(" ")
  #print(words[0:100])
  cnts_all = Counter()
  for word in words:
    #print(word)
    ngrams_word = ngrams(list(word), n_gram)
    
    #print(cnts)
    cnts_all.update(ngrams_word)
  
  
  return cnts_all
def get_ngrams_freqs_in_words(text_new,  n_gram):
  words = text_new.strip().split(" ")
  #print(words[0:100])
  cnts_all = Counter()
  for word in words:
    #print(word)
    ngrams_word = ngrams(list(word), n_gram)
    
    #print(cnts)
    cnts_all.update(ngrams_word)
  total =  sum(cnts_all.values())
  cnts = {k: v/total for k, v in cnts_all.items()}    
  
  return cnts  

def code_text(text, shifr_str, alphabet_str):
  trantab = text.maketrans(alphabet_str, shifr_str) 
  text = text.translate(trantab)
  return text  

def get_quality(original, decoded_text):
  equals = 0
  for i in range(len(original)):
    if list(original)[i] == list(decoded_text)[i]:
      equals += 1
  return (equals/len(original))  

def compare_freqs(bigrams1: dict, bigrams2 : dict) -> pd.DataFrame:
  data = pd.DataFrame(columns = ['bigram', 'corpus_freq', 'text_freq'])
  bigrams_list =  sorted(bigrams2.keys(), key=lambda bigram: bigrams2[bigram], reverse=True)
  bigram1_freqs = [bigrams1[bigram] for bigram in bigrams_list]
  bigram2_freqs = [bigrams2[bigram] for bigram in bigrams_list]
  data['bigram'] = bigrams_list
  data['corpus_freq'] = bigram1_freqs
  data['text_freq'] = bigram2_freqs
  return data


def change_shifr(shifr_str):
  shifr = list(shifr_str)
  ind1 = random.randint(0, len(shifr) - 1)
  ind2 = random.randint(0, len(shifr) - 1)
  #print(ind1, ind2)
  shifr[ind1],  shifr[ind2] = shifr[ind2],  shifr[ind1]
  return "".join(shifr)



def decode_text(coded_text, shifr_str, alphabet_str):
   trantab = coded_text.maketrans(shifr_str, alphabet_str) 
   decoded_text = coded_text.translate(trantab)
   return decoded_text  


def change_letters(alpha_str, l1, l2):
  alpha_list = list(alpha_str)
  ind1 = alpha_list.index(l1)
  ind2 = alpha_list.index(l2)
  alpha_list[ind1], alpha_list[ind2] =alpha_list[ind2], alpha_list[ind1]
  return "".join(alpha_list)   


def change_shifr_index(shifr_str, ind1, ind2):
  shifr_str_copy = shifr_str
  shifr = list(shifr_str_copy)
  shifr[ind1],  shifr[ind2] = shifr[ind2],  shifr[ind1]
  return "".join(shifr)  

def get_list_of_best_qualities(original, n_gram, amount):
  result = []
  freqs_large = get_ngrams_quantities(text_new,  n_gram)
  alphabet=list(set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя '))
  shifr  = alphabet.copy()
  random.shuffle(shifr)
  alphabet_str = ''.join(alphabet)
  shifr_str = ''.join(shifr)
  alphabet_str_init = alphabet_str
  shifr_str_init = shifr_str

  coded = code_text(original, shifr_str, alphabet_str)
  
  chelovecki_dict, _ =   get_letters_freqs(coded)
  chelovecki_letters = sorted(chelovecki_dict.keys(), key=lambda letter: chelovecki_dict[letter], reverse=True)
  chelovecki_letters.extend(letters_dict.keys() - chelovecki_dict.keys())
  chelovecki_letters_str = "".join(chelovecki_letters)

  for s  in range(amount):
    shifr_str = chelovecki_letters_str
    best_shifr = chelovecki_letters_str
    #best_result =  0
    best_score = -100000
    shifrs = []

    for i in range(10000):
      #print(shifr_str)
      #print(alph_letters_str)
      score_cur =  get_log_score_shifr(freqs_large, coded, shifr_str, alph_letters_str, n_gram)
      shifr_str_new = change_shifr(shifr_str)
      score_new = get_log_score_shifr(freqs_large, coded, shifr_str_new, alph_letters_str, n_gram)
      if score_new == score_cur:
        continue
      if (random.random() < np.exp(score_new - score_cur)): 
      #if (score_new > score_cur): 
        shifrs.append(shifr_str)
        shifr_str = shifr_str_new
        decoded_text = decode_text(coded, shifr_str, alph_letters_str)
        #print(decoded_text)
        #print(shifr_str_new)
        #print(score_new)
        if best_score < score_cur:
          best_score = score_cur
          best_shifr = shifr_str_new

    decoded_text = decode_text(coded, best_shifr, alph_letters_str)
    print(decoded_text)
    score = (get_quality(original, decoded_text))
    print(score)
    result.append(score)
  return (result)
  


def get_little_permutations():
  #alph_letters_str = change_letters(alph_letters_str, 'а', 'и')
  #alph_letters_str = change_letters(alph_letters_str, 'а', 'т')
  #alph_letters_str = change_letters(alph_letters_str, 'т', 'а')
  #alph_letters_str = change_letters(alph_letters_str, 'в', 'н')
  #alph_letters_str = change_letters(alph_letters_str, 'к', 'м')
  #alph_letters_str = change_letters(alph_letters_str, 'к', 'д')
  #alph_letters_str = change_letters(alph_letters_str, 'ь', 'я')
  #alph_letters_str = change_letters(alph_letters_str, 'о', 'и')
  #alph_letters_str = change_letters(alph_letters_str, 'в', 'т')
  #alph_letters_str = change_letters(alph_letters_str, 'о', 'е')
  alph_letters_str = alph_letters_str_orig
  prev_score = 1
  best_score = 0
  n = 0
  s = 0
  while prev_score != best_score:
    prev_score = best_score
    for i in range(1, 10):
      for j in range(1, 33 - i):
        alph_letters_str_new = change_shifr_index(alph_letters_str, j, j + i)
        decoded_text = decode_text(coded_text[n],  text_letters_str[n], alph_letters_str_new)
        score = (get_log_score(bigrams_quantity, decoded_text, 2))
        if score > best_score:
          print(score)
          print(decoded_text)
          best_score = score
          alph_letters_str = alph_letters_str_new

          print(alph_letters_str)
          print(decode_text(text_letters_str[n], shifr_str, alphabet_str))
          print(get_quality(text[n], decoded_text))
        s += 1  
  print(s) 