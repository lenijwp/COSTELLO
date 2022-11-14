import nltk
import sys

import os
sys.path.append('/data/jwp/codes/Tools/NL-Augmenter')
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
from nlaugmenter.transformations.tense.transformation import TenseTransformation
from nlaugmenter.transformations.gender_swap.transformation import GenderSwap
from nlaugmenter.transformations.synonym_substitution import SynonymSubstitution
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
import copy
import string
import random
sys.path.append(os.path.dirname(__file__))

class SentenceGenerator(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def changeTense(inputs:str):
        '''
        function: change the tense for inputs
        return: [output str]
        '''
        t=TenseTransformation('random')
        results=t.generate(inputs)
     
        if len(results)>0 and results[0]!=inputs:
            return [results[0]]
        return None

    @staticmethod
    def swapGender(inputs:str):
        '''
        function: change the tense for inputs
        return: [output str]
        '''
        t=GenderSwap()
        results=t.generate(inputs)
     
        if len(results)>0 and results[0]!=inputs:
            return [results[0]]
        return None
    
    @staticmethod
    def synonymSubstitution(inputs:str):
        t=SynonymSubstitution(max_output=3)
        results=t.generate(inputs)
        if len(results)>0:
            return results
        return None
    
    @staticmethod
    def getSynAntWord(word):
        '''
        function: get synonyms and antonyms for the given word by WordNet
        return: two str list
        '''
        syn = list()
        ant = list()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                syn.append(lemma.name())    #add the synonyms
                if lemma.antonyms():    #When antonyms are available, add them into the list
                    ant.append(lemma.antonyms()[0].name())
        return list(set(syn)),list(set(ant))

    @staticmethod
    def getWordSenti(word):
        '''
        function: get the sentiment label for a given word:
        return: one in ['obj','pos','neg',None]
        '''
        Analysis=list(swn.senti_synsets(word))
        if len(Analysis)>0:
            #word_state=swn.senti_synset(Analysis[0])
            word_state=Analysis[0]
            if word_state.obj_score()>0.5:
                return 'obj'
            else:
                return 'pos' if (word_state.pos_score()>word_state.neg_score()) else 'neg'
        return None

    @staticmethod
    def TokenSubstitute(alltokens:list,oldtoken:str,newtoken:str):
        '''
        function: substitute the token and merge all tokens to a sentence
        return : a str
        '''
        tmp=copy.deepcopy(alltokens)
        for i in range(len(tmp)):
            if tmp[i]==oldtoken:
                tmp[i]=newtoken
        return ' '.join(tmp)

    
            
    @staticmethod
    def changePunc(inputs:str):
        '''
        function: . <-> ! 
        '''
        token_word=word_tokenize(inputs)
        flag=False
        for i in range(len(token_word)):
            if token_word[i] =='.':
                flag=True
                token_word[i]='!'
                continue
            if token_word[i] =='!':
                flag=True
                token_word[i]='.'
        if flag == True:
            return [' '.join(token_word)]
        else:
            return None
    
    @staticmethod
    def sentiSynonymSubstitution(inputs:str,possi=0.5):

        token_word=word_tokenize(inputs)
        token_pos=pos_tag(token_word)
        stop_words = set(stopwords.words('english'))
        Results=[]  # store the results
        reverseMap={'pos':'neg','neg':'pos'}
        flag=0
        pos=-1
        for word,tag in token_pos:
            pos+=1
            ss_set=None
            if word in stop_words:
                continue
            if ('JJ' in tag or 'VB' in tag):
                senti_label=SentenceGenerator.getWordSenti(word)
                if (senti_label == None) or (senti_label == 'obj'):
                    continue
                syns,_=SentenceGenerator.getSynAntWord(word)
                for wd in syns:
                    if wd != word and (senti_label==SentenceGenerator.getWordSenti(wd)):
                        if random.uniform(0,1)<=possi:
                            token_word[pos]=wd
                            flag=1
        if flag==0:
            return None
        return [' '.join(token_word)]
    
### following is SinglePropertyTesting
    
    # @staticmethod
    # def SenWordFuzz(inputs:str):
    #     '''
    #     function: for each word with sentiment, get one synonyms vision, one antonyms vision.
    #     params inputs: a string as input
    #     return: a list, each element in the list is a tuple as (seed sentence, close sentence, far sentence)
    #     '''
    #     #print("inputs: ",inputs)
    #     token_word=word_tokenize(inputs)
    #     token_pos=pos_tag(token_word)
    #     stop_words = set(stopwords.words('english'))
    #     Results=[]  # store the results
        
    #     reverseMap={'pos':'neg','neg':'pos'}
    #     for word,tag in token_pos:
    #         ss_set=None
    #         if word in stop_words:
    #             continue
    #         if ('JJ' in tag or 'VB' in tag):
    #             senti_label=SentenceGenerator.getWordSenti(word)
    #             if (senti_label == None) or (senti_label == 'obj'):
    #                 continue
    #             syns,ants=SentenceGenerator.getSynAntWord(word)
    #             synword=None
    #             antword=None
    #             for wd in syns:
    #                 if wd != word and (senti_label==SentenceGenerator.getWordSenti(wd)):
    #                     synword=wd
    #                     break
    #             for wd in ants:
    #                 if wd != word and (reverseMap[senti_label]==SentenceGenerator.getWordSenti(wd)):
    #                     antword=wd
    #                     break
                        
    #             if None in [synword,antword]:
    #                 continue

    #             Results.append((inputs,SentenceGenerator.TokenSubstitute(token_word,word,synword),SentenceGenerator.TokenSubstitute(token_word,word,antword)))
    #     #print("results:",Results)
    #     return Results

    # @staticmethod
    # def SenWordFuzz(inputs:str):
    #     '''
    #     function: for each word with sentiment, get one synonyms vision, one antonyms vision.
    #     params inputs: a string as input
    #     return: a list, each element in the list is a tuple as (seed sentence, close sentence, far sentence)
    #     '''
    #     #print("inputs: ",inputs)
    #     token_word=word_tokenize(inputs)
    #     token_pos=pos_tag(token_word)
    #     stop_words = set(stopwords.words('english'))
    #     Results=[]  # store the results
        
    #     reverseMap={'pos':'neg','neg':'pos'}
    #     for word,tag in token_pos:
    #         ss_set=None
    #         if word in stop_words:
    #             continue
    #         if ('JJ' in tag or 'VB' in tag):
    #             senti_label=SentenceGenerator.getWordSenti(word)
    #             if (senti_label == None) or (senti_label == 'obj'):
    #                 continue
    #             syns,ants=SentenceGenerator.getSynAntWord(word)
    #             synword=[]
    #             antword=[]
    #             for wd in syns:
    #                 if wd != word and (senti_label==SentenceGenerator.getWordSenti(wd)):
    #                     synword.append(wd)
    #                     # break
    #             for wd in ants:
    #                 if wd != word and (reverseMap[senti_label]==SentenceGenerator.getWordSenti(wd)):
    #                     antword.append(wd)
    #                     # break
                        
    #             if len(synword)==0 or len(antword)==0:
    #                 continue
                
    #             for x in synword:
    #                 for y in antword:
    #                     Results.append((inputs,SentenceGenerator.TokenSubstitute(token_word,word,x),SentenceGenerator.TokenSubstitute(token_word,word,y)))
    #     #print("results:",Results)
    #     return Results
    
    @staticmethod
    def SenWordFuzz(inputs:str):
        '''
        function: for each word with sentiment, get one synonyms vision, one antonyms vision.
        params inputs: a string as input
        return: a list, each element in the list is a tuple as (seed sentence, close sentence, far sentence)
        '''
        #print("inputs: ",inputs)
        token_word=word_tokenize(inputs)
        token_pos=pos_tag(token_word)
        stop_words = set(stopwords.words('english'))
        Results=[]  # store the results
        
        reverseMap={'pos':'neg','neg':'pos'}
        for word,tag in token_pos:
            ss_set=None
            if word in stop_words:
                continue
            if ('JJ' in tag or 'VB' in tag):
                senti_label=SentenceGenerator.getWordSenti(word)
                if (senti_label == None) or (senti_label == 'obj'):
                    continue
                syns,ants=SentenceGenerator.getSynAntWord(word)

                synword=None
                antword=None
                for wd in syns:
                    if wd != word and (senti_label==SentenceGenerator.getWordSenti(wd)):
                        synword=wd
                        break
                for wd in ants:
                    if wd != word and (reverseMap[senti_label]==SentenceGenerator.getWordSenti(wd)):
                        antword=wd
                        break

                if None in [synword,antword]:
                    continue

                Results.append((inputs,SentenceGenerator.TokenSubstitute(token_word,word,synword),SentenceGenerator.TokenSubstitute(token_word,word,antword)))


        #print("results:",Results)
        return Results

    @staticmethod
    def cutDisjunc(inputs:str):
        '''
        function: cut the sentence at the position of disjunctive conjunction
        '''
        disjuncwordlist=['but','however','yet']
        token_word=word_tokenize(inputs)
        try:
            for i in range(len(token_word)):
                if token_word[i] in disjuncwordlist:
                    #return [(inputs,' '.join(token_word[:i]),' '.join(token_word[i+1:]))]
                    return [(inputs,' '.join(token_word[i+1:]),' '.join(token_word[:i]))]
        except:
            pass
        return None

if __name__ == "__main__":
    SA_generator=SentenceGenerator()
    print(SA_generator.SenWordFuzz('If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .'))
    print(SA_generator.changeTense('I have a good day.'))
    # print(SA_generator.changePunc('I have a good day.'))
    # print(SA_generator.cutDisjunc('I lost the game still it\'s been a good experience.'))
    # print(SA_generator.sentiSynonymSubstitution('I lost the game still it\'s been a good experience.'))
    # print(SA_generator.changeTense('I have a good day.'))
    # print(SA_generator.swapGender('He has a good day.'))