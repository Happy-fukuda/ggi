#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import pickle
from os import path
#発話
from gcp_texttospeech.srv import TTS
#音声認識
import stt_v
#word2vec(どのくらい似てるか)
import gensim.downloader as api
import sys
from ggi.srv import GgiLearning
from ggi.srv import GgiLearningResponse
import rospy

file_place='/home/athome/catkin_ws/src/ggi'


class ggitest():
    def __init__(self):
        #ベクトル読み込み

        self.word_vectors = api.load("glove-wiki-gigaword-100")
        print('ready to server')
        rospy.wait_for_service('/tts')
        self.tts=rospy.ServiceProxy('/tts', TTS)
        self.server=rospy.Service('/test_phase',GgiLearning,self.main)

    def main(self,req):
        if not path.isfile(file_place+'/object/object_file.pkl'):
            print('not found object file')
            sys.exit()

        else:
            with open(file_place+'/object/object_file.pkl','rb') as f:
                self.dict=pickle.load(f)

        self.tts("I await your instruction")
        while 1:

            name=[]
            name_feature=[]
            place=[]
            place_feature=[]
            string=stt_v.google_speech_api()

            shut='shut down'

            if  shut in string:
                self.tts("shut down")
                break

            else:
                i=0
                split=nltk.word_tokenize(string)
                for h in range(len(split)):
                    if split[h]=='the':
                        split[h]='a'
                pos = nltk.pos_tag(split)
                #場所とオブジェクトそれぞれの特徴と名前をいつにまとめる
                while i<len(pos):
                    if pos[i][1] =='IN' and pos[i][0]!='of':

                        for j in range(i,len(pos)):
                            if 'NN' in pos[j][1]:
                                place.append(pos[j][0])
                                if j!=(len(pos)-1):
                                    if not 'NN' in pos[j+1][1]:
                                        break
                            elif pos[j][1]=='JJ':
                                place_feature.append(pos[j][0])
                        i=j+1
                        continue

                    elif pos[i][1] =='IN' and pos[i][0]=='of':

                        for k in range(i,len(pos)):
                            if 'NN' in pos[k][1]:
                                name.append(pos[k][0])
                                if k!=(len(pos)-1):
                                    if not 'NN' in pos[k+1][1]:
                                        break
                            elif pos[k][1]=='JJ':
                                name_feature.append(pos[k][0])
                        i=k+1
                        continue

                    elif 'NN' in pos[i][1]:
                        name.append(pos[i][0])

                    elif pos[i][1]=='JJ':
                        name_feature.append(pos[i][0])
                    i+=1
                print(name)
                print(place)
                print(name_feature)
                print(place_feature)



                str=self.branch(name,name_feature,place,place_feature)
                if str=='no':
                    self.tts('one more time')
                    continue

                elif str:
                    return GgiLearningResponse(location_name=str)
                else:
                    self.tts("I don't know " )
                    self.tts('one more time')

    def branch(self,name,name_feature,place,place_feature):
        str=''
        defalt=0 #value用
        correct=0 #要素数
        succese=False

        long=len(self.dict['place_name'])
        for i in range(long):

        #優先度順に確認
            if set(name) & set(self.dict['object_name'][i]) and set(place) & set(self.dict['place_name'][i]): #オブジェクトの名前と場所
                self.tts('I will go '+' '.join(self.dict['place_feature'][i]) +' '.join(self.dict['place_name'][i])+' is this  OK?')
                while 1:
                    y=stt_v.google_speech_api()
                    if 'yes' in y:
                        self.tts('OK.')
                        break
                    elif 'no' in y:
                        return 'no'
                if self.dict['place_feature'][i]:
                    str=' '.join(self.dict['place_feature'][i]) +' '+' '.join(self.dict['place_name'][i])
                else:
                    str=' '.join(self.dict['place_name'][i])

                print('pla + feature')
                return str

            if set(place) & set(self.dict['place_name'][i]) and set(place_feature) & set(self.dict['place_feature'][i]): #場所と場所の特徴
                self.tts('I will go '+' '.join(self.dict['place_feature'][i]) +' '.join(self.dict['place_name'][i])+' is this  OK?')
                while 1:
                    y=stt_v.google_speech_api()
                    if 'yes' in y:
                        self.tts('OK.')
                        break
                    elif 'no' in y:
                        return 'no'
                if self.dict['place_feature'][i]:
                    str=' '.join(self.dict['place_feature'][i]) +' '+' '.join(self.dict['place_name'][i])
                else:
                    str=' '.join(self.dict['place_name'][i])
                print('pl + feature')
                return str

            try:
                for na in place:
                    for ob_na in self.dict['place_name'][i]:
                        value= self.word_vectors.similarity(na,ob_na)
                        if value>0.6 and set(place_feature) & set(self.dict['place_feature'][i]):
                            self.tts('I will go '+' '.join(self.dict['place_feature'][i]) +' '.join(self.dict['place_name'][i])+' is this  OK?')
                            while 1:
                                y=stt_v.google_speech_api()
                                if 'yes' in y:
                                    self.tts('OK.')
                                    break
                                elif 'no' in y:
                                    return 'no'
                            if self.dict['place_feature'][i]:
                                str=' '.join(self.dict['place_feature'][i]) +' '+' '.join(self.dict['place_name'][i])
                            else:
                                str=' '.join(self.dict['place_name'][i])

                            print('wo2 + pl')

                            return str
            except:
                pass

            if set(name) & set(self.dict['object_name'][i]) and set(name_feature) & set(self.dict['object_feature'][i]): #オブジェクトの名前と特徴
                self.tts('I will go '+' '.join(self.dict['place_feature'][i]) +' '.join(self.dict['place_name'][i])+' is this  OK?')
                while 1:
                    y=stt_v.google_speech_api()
                    if 'yes' in y:
                        self.tts('OK.')
                        break
                    elif 'no' in y:
                        return 'no'
                if self.dict['place_feature'][i]:
                    str=' '.join(self.dict['place_feature'][i]) +' '+' '.join(self.dict['place_name'][i])
                else:
                    str=' '.join(self.dict['place_name'][i])
                print('name and feature')

                return str


        try:
            for na in place:
                for ob in range(long):
                    for ob_na in self.dict['place_name'][ob]:
                        value= self.word_vectors.similarity(na,ob_na)
                        if value>0.5 and defalt<value:
                            defalt=value
                            correct=ob
                            succese=True
            if succese:
                self.tts('I will go '+' '.join(self.dict['place_feature'][correct]) +' '.join(self.dict['place_name'][correct])+' is this  OK?')
                while 1:
                    y=stt_v.google_speech_api()
                    if 'yes' in y:
                        self.tts('OK.')
                        break
                    elif 'no' in y:
                        return 'no'
                if self.dict['place_feature'][correct]:
                    str=' '.join(self.dict['place_feature'][correct]) +' '+' '.join(self.dict['place_name'][correct])
                else:
                    str=' '.join(self.dict['place_name'][correct])
                print('word pl')

                return str
        except:
            pass


        try:
            for na in name:
                for ob in range(long):
                    for ob_na in self.dict['object_name'][ob]:
                        value= self.word_vectors.similarity(na,ob_na) #オブジェクトの名前がどのぐらい似てるか
                        if value>0.5 and defalt<value:
                            defalt=value
                            correct=ob
                            succese=True
            if succese:
                self.tts('I will go '+' '.join(self.dict['place_feature'][correct]) +' '.join(self.dict['place_name'][correct])+' is this  OK?')
                while 1:
                    y=stt_v.google_speech_api()
                    if 'yes' in y:
                        self.tts('OK.')
                        break
                    elif 'no' in y:
                        return 'no'
                if self.dict['place_feature'][correct]:
                    str=' '.join(self.dict['place_feature'][correct]) +' '+' '.join(self.dict['place_name'][correct])
                else:
                    str=' '.join(self.dict['place_name'][correct])
                print('word ob')

                return str
        except:
            pass
        return str





if __name__=='__main__':
    rospy.init_node('ggi_test')
    ggi=ggitest()
    rospy.spin()
