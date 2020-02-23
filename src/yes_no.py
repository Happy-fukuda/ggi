#!/usr/bin/env python
# -*- cording: utf-8 -*-

#発話
from gcp_texttospeech.srv import TTS
#音声認識
import stt

import rospy
from std_msgs.msg import String
from ggi.srv import YesNo
from ggi.srv import YesNoResponse

answer_list=['yes','no']

class GgiinStruction:
    def __init__(self):

        print("server is ready")
        self.server=rospy.Service('/yes_no',YesNo,self.yes_no)
        self.tts=rospy.ServiceProxy('/tts', TTS)
    def yes_no(self,req):
        while 1:
            str=stt.google_speech_api(phrases=answer_list)
            if answer_list[0] in str:
                return YesNoResponse(result=True)
            elif answer_list[1] in str:
                return YesNoResponse(result=False)
            else:
                pass


if __name__=='__main__':
    rospy.init_node("yes_no")
    GgiinStruction()
    rospy.spin()
