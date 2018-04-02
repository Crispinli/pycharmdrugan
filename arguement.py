from PIL import Image
import os

root_A = "./input/monet2photo/trainA"
root_B = "./input/monet2photo/trainB"

filenames_A = os.listdir(root_A)
filenames_B = os.listdir(root_B)

count = 0
region = (42, 42, 214, 214)

# for name in filenames_A:
#     path_A = os.path.join(root_A, name)
#     img = Image.open(path_A).resize([256, 256])
#     # img_1 = img.rotate(45).crop(region).resize(img.size)
#     # img_2 = img.rotate(-45).crop(region).resize(img.size)
#     img_1 = img.rotate(45).resize(img.size)
#     img_2 = img.rotate(-45).resize(img.size)
#     img_3 = img.transpose(Image.FLIP_LEFT_RIGHT)
#     if not os.path.exists("./input/monet2photo/train_A/"):
#         os.makedirs("./input/monet2photo/train_A/")
#     img.save("./input/monet2photo/train_A/" + str(count) + "_0.jpg")
#     img_1.save("./input/monet2photo/train_A/" + str(count) + "_1.jpg")
#     img_2.save("./input/monet2photo/train_A/" + str(count) + "_2.jpg")
#     img_3.save("./input/monet2photo/train_A/" + str(count) + "_3.jpg")
#     count += 1
#     print(count)

# count = 0

for name in filenames_B:
    path_B = os.path.join(root_B, name)
    img = Image.open(path_B).resize([256, 256])
    # img_1 = img.rotate(45).crop(region).resize(img.size)
    # img_2 = img.rotate(-45).crop(region).resize(img.size)
    # img_1 = img.rotate(45).resize(img.size)
    # img_2 = img.rotate(-45).resize(img.size)
    # img_3 = img.transpose(Image.FLIP_LEFT_RIGHT)
    if not os.path.exists("./input/monet2photo/train_B/"):
        os.makedirs("./input/monet2photo/train_B/")
    img.save("./input/monet2photo/train_B/" + str(count) + ".jpg")
    # img.save("./input/monet2photo/train_B/" + str(count) + "_0.jpg")
    # img_1.save("./input/monet2photo/train_B/" + str(count) + "_1.jpg")
    # img_2.save("./input/monet2photo/train_B/" + str(count) + "_2.jpg")
    # img_3.save("./input/monet2photo/train_B/" + str(count) + "_3.jpg")
    count += 1
    print(count)
