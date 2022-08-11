#学生：何迅
#创建时间：2022/7/11 19:19
import pymysql

__config__ = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "ocr"
}

try:
    db = pymysql.connect(
        **__config__
    )
except Exception as e:
    print(e)