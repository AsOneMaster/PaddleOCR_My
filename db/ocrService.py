#学生：何迅
#创建时间：2022/7/11 21:17
from db.ocrDao import OCRDao


class OcrService:
    # 创建私有对象
    __ocr_dao = OCRDao()

    # 插入检测结果函数
    def insert_ocr(self, res, path):
        result = self.__ocr_dao.insert_ocr(res, path)
        return result

    # 查询图片，是否已经检测
    def select_ocr(self, path):
        result = self.__ocr_dao.select_ocr(path)
        return result

    # 分页查询最近记录
    def select_limit(self, page):
        result = self.__ocr_dao.select_limit(page)
        return result

    # 查询所有数据
    def select_all(self):
        result = self.__ocr_dao.select_all()
        return result

    # 查询记录数量
    def select_page(self):
        result = self.__ocr_dao.select_page()
        return result

    # 更新表格数据
    def update_page(self, value, id):
        self.__ocr_dao.update_page(value, id)
    # 删除记录
    def delete_row(self, id):
        self.__ocr_dao.delete_row(id)
    # # 查询用户角色
    # def search_user_role(self, username):
    #     role = self.__user_dao.search_user_role(username)
    #     return role