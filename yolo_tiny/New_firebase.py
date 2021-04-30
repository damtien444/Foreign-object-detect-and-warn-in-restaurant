import pyrebase
from datetime import datetime

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
#init
firebase=pyrebase.initialize_app(firebaseConfig)
db=firebase.database()
#auth=firebase.auth()
storage=firebase.storage()
#filename=input("Nhap ten anh")
#cloudfilename=input("Nhap link cloud")
#storage.child(cloudfilename).put(filename)

#print(storage.child(cloudfilename).get_url(None))

#id
ref = db.child("index").get()
i=ref.val()
i=i+1
#time
now = datetime.now()
print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%H:%M:%S %d/%m/%Y ")
print("Ngay va gio hien tai =", dt_string)

#data
data={
    'id':i,
    'name':"VU QUEN DO SO "+str(i),
    'hasMessage':0,
    'date':dt_string,
    'image':"https://firebasestorage.googleapis.com/v0/b/vdk18n10.appspot.com/o/images%2Ffoo.jpg?alt=media"
}
db.child("FORGETTHING").push(data)
db.child("index").set(i)

push_service = FCMNotification(api_key="AAAACnosiMA:APA91bEqyBuwY_7H6V6MwtDUpMIx3XhIC3pKJYtMPKQF8CsdRbXIEjiOIt-8AxLwC8uDLKKjiXQXxP44QFlMZENKnxxdAqUZ7bbkqjt49QSy0lEvTEXjrvu_pzLPpWaj_4w-SSf6RAMo")


# Send to multiple devices by passing a list of ids.
registration_ids = ["fzrgkY3cQ8yjjjR8auxYoD:APA91bH65N-7u4FmYKy61j87lBtExFOXInZhIgeh4ztdv7WJTlGkpjaCGjdPS1X9plmf53GXWCv5tI_9_7D4nZ1uysw4wfzA5SXSjVYHmimI6pnCxLRoSPJIWZTm7oiAou-_O4sIctTJ"]
message_title = "Thông báo có người để quên "
message_body = "Chào anh, có người để quên đồ trong cửa hàng mau mau nhắc khách đi nhá "
result = push_service.notify_multiple_devices(registration_ids=registration_ids, message_title=message_title, message_body=message_body)

print (result)