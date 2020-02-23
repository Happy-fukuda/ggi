#!/usr/bin/env python
# -*- cording: utf-8 -*-

# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Google Cloud Text-To-Speech API sample application .

Example usage:
    python quickstart.py
'''
import sys
import pyaudio
import wave
import time
import rospy
from ggi.srv import TTS
from ggi.srv import TTSResponse




class call:
    def __init__(self):
        rospy.init_node("tts_srv",anonymous=True)
        print("server is ready")
        server=rospy.Service('/tts',TTS,self.run_quickstart)

    def run_quickstart(self,req):


        # [START tts_quickstart]
        '''Synthesizes speech from the input string of text or ssml.

        Note: ssml must be well-formed according to:
        https://www.w3.org/TR/speech-synthesis/
        '''
        from google.cloud import texttospeech


        try:

            # Instantiates a client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            #print(goal.string_data)
            synthesis_input = texttospeech.types.SynthesisInput(text=req.str)

            # Build the voice request, select the language code ("en-US") and the ssml
            # voice gender ("neutral")
            voice = texttospeech.types.VoiceSelectionParams(
                language_code='en-US',
                ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

             #Select the type of audio file you want returned
            audio_config = texttospeech.types.AudioConfig(
                        audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
            response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # The response's audio_content is binary.
            with open("output.wav", 'wb') as out:
        # Write the response to the output file.
                out.write(response.audio_content)

            Filename="output.wav"

            wf = wave.open(Filename, 'rb')

            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

            chunk = 1024
            data = wf.readframes(chunk)
            while len(data)>0:
                stream.write(data)
                data = wf.readframes(chunk)

            stream.stop_stream()
            stream.close()
            p.terminate()

            return TTSResponse()
        except Exception as e:
            print ("texttospeech failed : %s",e)
            return TTSResponse()

if __name__=='__main__':
    call()
    rospy.spin()
