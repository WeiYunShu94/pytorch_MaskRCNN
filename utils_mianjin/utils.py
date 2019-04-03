import os

class UTILS(object):
    def __init__(self):
        pass

    def rename(self,img_file):
        """
        将一个文件夹下的所有图片，按照顺序从1到i进行命名
        """
        img_list = os.listdir(img_file)
        for i,img_name in enumerate(img_list):
            new_item = str(i+1)+'.jpg'
            os.rename(os.path.join(img_file,img_name),os.path.join(img_file,new_item))

if __name__=="__main__":
    UTILS.rename()