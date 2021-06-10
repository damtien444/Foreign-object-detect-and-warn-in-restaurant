from datetime import datetime

import pyrebase
from pyfcm import FCMNotification

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

def converter(file,output):
    return open(output, "wb").write(open(file, "rb").read())

class Firebase:
    def __init__(self):
        converter("video/ras.avi", 'video/ras.mp4')
        self.firebase=pyrebase.initialize_app(firebaseConfig)
        self.db = self.firebase.database()
        self.storage = self.firebase.storage()
        self.now = datetime.now()
    def __getid(self):
        re = self.db.child("index").get()
        return re.val() + 1
    def __gettime(self):
        print("now =", self.now)
        # dd/mm/YY H:M:S
        dt_string = self.now.strftime("%H:%M:%S %d/%m/%Y ")
        print("Ngay va gio hien tai =", dt_string)
        return dt_string
    def __getweekday(self):
        week = ['hai', 'ba', 'tư', 'năm', 'sáu', 'bảy', 'Chủ nhật']
        d, m, y = self.now.strftime("%d %m %Y ").split()
        wd = datetime(int(y), int(m), int(d))
        return week[wd.weekday()]
    def push_data(self):
        i = self.__getid()
        self.storage.child("images/ras" + str(self.now) + ".png").put('./image/ras.png')
        self.storage.child("videos/ras" + str(self.now) + ".mp4").put('./video/ras.mp4')
        data = {
            'id': i,
            'name': "VỤ QUÊN ĐỒ NGÀY THỨ " + self.__getweekday(),
            'hasMessage': 0,
            'date': self.__gettime(),
            'images': self.storage.child("images/ras" + str(self.now) + ".png").get_url(None)
        }
        self.db.child("FORGETTHING").push(data)
        self.db.child("index").set(i)
        push_service = FCMNotification(api_key="AAAACnosiMA:APA91bEqyBuwY_7H6V6MwtDUpMIx3XhIC3pKJYtMPKQF8CsdRbXIEjiOIt-8AxLwC8uDLKKjiXQXxP44QFlMZENKnxxdAqUZ7bbkqjt49QSy0lEvTEXjrvu_pzLPpWaj_4w-SSf6RAMo")
        registration_ids = ["dC8ej3vBSRmSn48yH31f9u:APA91bFjHLUqS-hJ7JHt0gTvuWQAK_mGKS6fLmkJFiOf1GifO3z0b1GR7HjLSfJ12A6542X3ZVXBtUQIYYf8Hul3YQbYPweJfMwULqVE-Z7EmPhcSOF_wD84jrpapNjfxGW9eGVUMibW"]
        message_title = "Thông báo có người để quên "
        message_body = "Chào anh, có người để quên đồ trong cửa hàng mau mau nhắc khách đi nhá "
        result = push_service.notify_multiple_devices(registration_ids=registration_ids, message_title=message_title,
                                                      message_body=message_body)
        print(result)
        print(data)



if __name__ == '__main__':
    # converter("video/ras.avi", 'video/ras.mp4')
    fb = Firebase()
    fb.getdata()