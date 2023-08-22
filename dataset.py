from PIL import Image
import os

# div2k_dir = os.listdir("E:/Ds/Project/dataset/DIV2K/DIV2K_train_HR")
# index = 0
# for i in div2k_dir:
#     img_path = os.path.join("E:/Ds/Project/dataset/DIV2K/DIV2K_train_HR/" + i)
#     img = Image.open(img_path)
#     width, height = img.size
#     if width >= 128 and height >=128:
#         img.save("E:/Ds/Project/dataset/DF2K_OST/train/DIV2K_{}.png".format(index),"PNG")
#         index += 1

# flickr2f_dir = os.listdir("E:/Ds/Project/dataset/Flickr2K/Flickr2K_HR")
# index = 0
# for i in flickr2f_dir:
#     img_path = os.path.join("E:/Ds/Project/dataset/Flickr2K/Flickr2K_HR/" + i)
#     img = Image.open(img_path)
#     width, height = img.size
#     if width >= 128 and height >=128:
#         img.save("E:/Ds/Project/dataset/DF2K_OST/train/Flickr2K_{}.png".format(index),"PNG")
#         index += 1

ost1 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/animal")
ost2 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/building")
ost3 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/grass")
ost4 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/mountain")
ost5 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/plant")
ost6 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/sky")
ost7 = os.listdir("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/water")

index = 0
for i in ost1:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/animal/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1

for i in ost2:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/building/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1

for i in ost3:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/grass/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1

for i in ost4:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/mountain/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1

for i in ost5:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/plant/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1

for i in ost6:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/sky/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1


for i in ost7:
    img_path = os.path.join("E:/Ds/Project/dataset/OutdoorSceneTrain_v2/water/" + i)
    img = Image.open(img_path)
    width, height = img.size
    if width >= 128 and height >=128:
        img.save("E:/Ds/Project/dataset/OST/train/ost_{}.png".format(index),"PNG")
        index += 1





