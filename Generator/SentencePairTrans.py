import os
import sys
sys.path.append(os.path.dirname(__file__))
from SentenceTrans import *
from SentenceTrans import SentenceGenerator as SentenceGenerator
from checklist.perturb import Perturb
import spacy
import random
spacy_nlp=spacy.load("en_core_web_sm")

class SentencePairGenerator(object):
    def __init__(self):
            pass

    @staticmethod
    def ChangePersonNamePair(s1,s2,max_output=1):
        perturbed = Perturb.perturb([spacy_nlp(s1)], Perturb.change_names, nsamples=1)
        perturbed_text_s1 = (
            perturbed.data[0][1 : ]
            if len(perturbed.data) > 0
            else None
        )
        s=s1+' [SEP] '+s2
        perturbed = Perturb.perturb([spacy_nlp(s)], Perturb.change_names, nsamples=1)
        perturbed_text_s = (
            perturbed.data[0][1 : ]
            if len(perturbed.data) > 0
            else None
        )
        if perturbed_text_s1 is None or perturbed_text_s is None:
            return None

        res=[]
        
        token_old=word_tokenize(s)
        for i in range(len(perturbed_text_s)):
            token_new=word_tokenize(perturbed_text_s[i])
            change=[None,None]
            pos=0
            if len(token_new)!=len(token_old):
                continue
            for j in range(len(token_new)):
                if token_new[j]=='SEP':
                    pos=1
                if token_new[j]!=token_old[j]:
                    change[pos]=token_old[j]
            if (None not in change) and (change[0]==change[1]):
                res.append((s1+' [SEP] '+s2,perturbed_text_s[i],random.choice(perturbed_text_s1)+' [SEP] '+s2))
                

        if len(res)==0:
            return None
        return random.sample(res, min(max_output,len(res)))
    
    @staticmethod
    def ChangeLocationPair(s1,s2,max_output=1):
        perturbed = Perturb.perturb([spacy_nlp(s1)], Perturb.change_location, nsamples=1)
        perturbed_text_s1 = (
            perturbed.data[0][1 : ]
            if len(perturbed.data) > 0
            else None
        )
        s=s1+' [SEP] '+s2
        perturbed = Perturb.perturb([spacy_nlp(s)], Perturb.change_location, nsamples=1)
        perturbed_text_s = (
            perturbed.data[0][1 : ]
            if len(perturbed.data) > 0
            else None
        )
        if perturbed_text_s1 is None or perturbed_text_s is None:
            return None

        res=[]
        
        token_old=word_tokenize(s)
        for i in range(len(perturbed_text_s)):
            token_new=word_tokenize(perturbed_text_s[i])
            change=[None,None]
            pos=0
            if len(token_new)!=len(token_old):
                continue
            for j in range(len(token_new)):
                if token_new[j]=='SEP':
                    pos=1
                if token_new[j]!=token_old[j]:
                    change[pos]=token_old[j]
            if (None not in change) and (change[0]==change[1]):
                res.append((s1+' [SEP] '+s2,perturbed_text_s[i],random.choice(perturbed_text_s1)+' [SEP] '+s2))
                

        if len(res)==0:
            return None
        return random.sample(res, min(max_output,len(res)))

    @staticmethod
    def SynAntSubstitute(s1,s2,max_output=1):
        inputs=s1+' [SEP] '+s2
        '''
        function: for each word with sentiment, get one synonyms vision, one antonyms vision.
        params inputs: a string as input
        return: a list, each element in the list is a tuple as (seed sentence, close sentence, far sentence)
        '''
        token_word=word_tokenize(inputs)
        token_pos=pos_tag(token_word)
        stop_words = set(stopwords.words('english'))
        Results=[]  # store the results
        
        for word,tag in token_pos:
            ss_set=None
            if word in stop_words:
                continue
            if ('JJ' in tag or 'VB' in tag):
                syns,ants=SentenceGenerator.getSynAntWord(word)
                synword=None
                antword=None
                for wd in syns:
                    if wd != word:
                        synword=wd
                        break
                for wd in ants:
                    if wd != word:
                        antword=wd
                        break
                        
                if None in [synword,antword]:
                    continue

                Results.append((inputs,SentenceGenerator.TokenSubstitute(token_word,word,synword),SentenceGenerator.TokenSubstitute(token_word,word,antword)))
        
        return Results

if __name__ == "__main__":
    SA_generator=SentencePairGenerator()
    print(SA_generator.ChangePersonNamePair('Alice likes Bob.','Bob is liked by Alice.'))
    print(SA_generator.ChangeLocationPair('Alice comes from Keller','Alice is from Keller'))
    print(SA_generator.SynAntSubstitute('Alice comes from Keller','Alice is from Keller'))