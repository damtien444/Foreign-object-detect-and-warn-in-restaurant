import glob

import cv2
import pyrebase
from playsound import playsound

firebaseConfig = {
    'apiKey': "AIzaSyCjJTiC2_Fa16wT21dgkh1_3_wJLEGQMME",
    'authDomain': "vdk18n10.firebaseapp.com",
    'databaseURL': "https://vdk18n10-default-rtdb.firebaseio.com",
    'projectId': "vdk18n10",
    'storageBucket': "vdk18n10.appspot.com",
    'messagingSenderId': "44999411904",
    'appId': "1:44999411904:web:4c6b714260e8be928b0525",
    'measurementId': "G-QSPLKRGD34"
}
#init

firebase=pyrebase.initialize_app(firebaseConfig)
db = firebase.database()



def PlaySound(path):
    for song in glob.glob(path):
        playsound(song)

def play():
    while True:
        users = db.child("FORGETTHING").get()
        if(users.each() is not None):
            for user in users.each():
                if user.val()['hasMessage'] == 1:
                    db.child("FORGETTHING").child(user.key()).update({'hasMessage': 0})
                    PlaySound("video/TunePocket-Trendy-Audio-Logo-Preview.mp3")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    play()
    # PlaySound("video/TunePocket-Trendy-Audio-Logo-Preview.mp3")

