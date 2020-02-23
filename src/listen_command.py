#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gcp_texttospeech.srv import TTS
#from std_srvs.srv import Empty
import stt

import Levenshtein as lev
import rospy
from ggi.srv import ListenCommand
from ggi.srv import ListenCommandResponse
#filename and path
file='/home/athome/catkin_ws/src/ggi/set_word/'




class GgiinStruction:
    def __init__(self):

        print("server is ready")
        rospy.wait_for_service('/tts')
        self.server=rospy.Service('/listen_command',ListenCommand,self.main)
        #self.sound=rospy.ServiceProxy('/sound', Empty)
        self.tts=rospy.ServiceProxy('/tts', TTS)


    def main(self,req):
        with open(file+req.file_name,'r') as f:
            speak_list=[line.strip() for line in f.readlines()]
        self.tts("ready")
        string=stt.google_speech_api(phrases=speak_list)
        num_lev=1
        prog_num=-1
        for str_num in range(len(speak_list)):
            result = lev.distance(string, speak_list[str_num])/(max(len(string), len(speak_list[str_num])) *1.00)

            if result <0.2:
                prog_num=str_num
                break

            elif num_lev>result and result<0.5:
                num_lev=result
                prog_num=str_num

        success=False
        if prog_num<0 or prog_num>=len(speak_list):
            command=''
            success=False
        else:
            command=speak_list[prog_num]
            success=True


        return ListenCommandResponse(cmd=command,result=success)



if __name__=='__main__':
    rospy.init_node('listen_command')

    ggi=GgiinStruction()
    rospy.spin()
