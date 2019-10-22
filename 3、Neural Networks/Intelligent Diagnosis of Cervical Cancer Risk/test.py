import os


def readname(filePath):
    filenames = os.listdir(filePath)
    for name in filenames:
        filenames[filenames.index(name)] = name[:-4]
    return filenames


if __name__ == "__main__":
    filePath = "pos_1/"
    scale = 20
    names = readname(filePath)
    for name in names:
        file1 = filePath + name + ".kfb"
        label1 = filePath + name + ".json"
        print(label1)
