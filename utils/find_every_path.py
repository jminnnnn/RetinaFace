import os, sys

class FindEveryPath():
    def __init__(self):
        self.paths = []
    
    def FindAll(self, path):
        if not os.path.isdir(path):
            return
        files = os.listdir(path)
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    self.FindAll(fullpath)
                else:
                    if "." in fullpath:
                        self.paths.append(fullpath)

    def get_paths(self):
        return self.paths

if __name__ == "__main__":
    tool = FindEveryPath()
    tool.FindAll("result")
    print(tool.get_paths())
