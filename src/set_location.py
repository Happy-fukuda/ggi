#!/usr/bin/env python
# -*- coding: utf-8 -*-

#発話
from gcp_texttospeech.srv import TTS
#音声認識
import stt

import nltk
import pickle
from os import path
import rospy
from ggi.srv import GgiLearning
from ggi.srv import GgiLearningResponse

place_file='/home/athome/catkin_ws/src/ggi/set_word/place_name'
file='/home/athome/catkin_ws/src/ggi/src/location.pkl' #作成場所の指定

class GgiinStruction:
    def __init__(self):
        with open(place_file,'r') as c:
            self.place_template=[line.strip() for line in c.readlines()]

        print("server is ready")
        rospy.wait_for_service('/tts')
        self.server=rospy.Service('/name_registration',GgiLearning,self.register_object)
        self.tts=rospy.ServiceProxy('/tts', TTS)


    #オブジェクト登録
    def register_object(self,req):
        recognition=''

        self.tts('Please tell me the place.')
        while 1:
            if not recognition:

                string=stt.google_speech_api(phrases=self.place_template)
                self.tts(string +' Is this OK?')

            recognition = stt.google_speech_api()

            if 'yes' in recognition:
                res=self.save_name(string , False)
                break

            elif 'no' in recognition:
                self.tts('please one more time')
                recognition=''

            elif 'again' in recognition:
                self.tts(string +' Is this OK?')

        #self.tts('I understand.')

        return GgiLearningResponse(location_name=res)



    #保存          s=string ob=name or place (True or False)
    def save_name(self,s):

        #初期化
        if not path.isfile(file) or path.getsize(file)==0:
            with open(file,"wb") as f:
                dictionary={'place_name':[],'place_feature':[]}
                pickle.dump(dictionary, f)

        pos = nltk.pos_tag(nltk.word_tokenize(s))
        feature=[]   #一時保存用
        name=[]     #一時保管用

        with open(file,'rb') as web:
            dict=pickle.load(web)


        with open(file,'wb') as f:
            for i in range(len(pos)):
                if pos[i][1]=='JJ':
                    feature.add(pos[i][0])

                elif 'NN' in pos[i][1]:
                    name.append(pos[i][0])

            dict['place_name'].append(name)
            dict['place_feature'].append(feature)
            pickle.dump(dict, f)
            if dict['place_feature'][len(dict['place_feature'])-1]:
                str='_'.join(dict['place_feature'][len(dict['place_feature'])-1])+'_'+'_'.join(dict['place_name'][len(dict['place_name'])-1])
            else:
                str='_'.join(dict['place_name'][len(dict['place_name'])-1])
            print(dict)
            return str





if __name__=='__main__':
    rospy.init_node('ggi_learning')
    GgiinStruction()
    rospy.spin()
