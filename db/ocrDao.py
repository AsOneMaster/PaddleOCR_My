#学生：何迅
#创建时间：2022/7/11 19:32
from db.mysql_db import db


class OCRDao:
    # 插入记录
    def insert_ocr(self, res, path):
        cursor = db.cursor()
        sql = "INSERT INTO ocr(`Ocr_Result`,`Save_Path`,`Ocr_Time`) VALUES (%s,%s,NOW())"
        try:

            cursor.execute(sql, (res, path))
            # result = cursor.fetchall()
            # print(result)
            db.commit()
            return 'sucess'
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao err:",e)
            return 'failed'
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 查询图片，是否已经检测
    def select_ocr(self, path):
        cursor = db.cursor()
        sql = "SELECT Save_Path FROM ocr WHERE Save_Path=%s"
        try:

            result = cursor.execute(sql, (path))
            # result = cursor.fetchall()
            # print(result)
            db.commit()
            print('---------------ocrDao索引:', result)
            return result
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 分页查询最近记录
    def select_limit(self, page):
        cursor = db.cursor()
        sql = "SELECT id,Ocr_Result,Save_Path,Ocr_Time FROM ocr ORDER BY id DESC LIMIT 5 OFFSET %s"
        try:

            cursor.execute(sql, (page*5))
            result = cursor.fetchall()
            # print(result)
            db.commit()
            print('---------------ocrDao最近记录:', result)
            return result
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao最近记录 err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 查询所有数据
    def select_all(self):
        cursor = db.cursor()
        sql = "SELECT Ocr_Result,Save_Path,Ocr_Time FROM ocr ORDER BY id DESC"
        try:

            cursor.execute(sql)
            result = cursor.fetchall()
            # print(result)
            db.commit()
            print('---------------ocrDao所有记录:', result[0])
            return result
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao所有记录 err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 查询记录数量
    def select_page(self):
        cursor = db.cursor()
        sql = "SELECT COUNT(id) FROM ocr"
        try:

            cursor.execute(sql)
            result = cursor.fetchall()
            # print(result)
            db.commit()
            result = int(result[0][0])   # 5条数据一页
            print('---------------ocrDao Page:', result)
            return int(result)
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao Page err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 更新记录
    def update_page(self, value, id):
        cursor = db.cursor()
        sql = "UPDATE ocr SET Ocr_Result = %s WHERE id = %s"
        try:

            result = cursor.execute(sql, (value, id))
            db.commit()
            print('---------------ocrDao Update:', result)
            return int(result)
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao Update err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()

    # 删除记录
    def delete_row(self, id):
        cursor = db.cursor()
        sql = "DELETE FROM ocr WHERE id = %s"
        try:

            result = cursor.execute(sql, (id))
            db.commit()
            print('---------------ocrDao Delete:', result)
            return int(result)
        except Exception as e:
            if "db" in dir():
                db.rollback()
            print("---------------ocrDao Delete err:", e)
            return None
        finally:
            if "db" in dir():
                cursor.close()
                db.close()
# cursor = db.cursor()
# sql = "UPDATE ocr SET " + 'Ocr_Result' + "= %s WHERE id = %s"
# try:
#
#     result = cursor.execute(sql, ('52', '26'))
#     # result = cursor.fetchall()
#     # print(result)
#     db.commit()
#     # result = int(result[0][0])  # 5条数据一页
#     print('---------------ocrDao Update:', result)
# except Exception as e:
#     if "db" in dir():
#         db.rollback()
#     print("---------------ocrDao Update err:", e)
# finally:
#     if "db" in dir():
#         cursor.close()
#         db.close()