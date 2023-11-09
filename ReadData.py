import pandas as pd

class ReadData:
    def __init__(self, filename):
        self.__filename = filename
    def loadfile(self):
        try:
            data = pd.read_csv(self.__filename)
            return data
        #When the filename in incorrect or file not in place
        except IOError as ioerr:
            print("File error: " + str(ioerr))
            print("Check the filename and put it in the program directory.")
        #When filename is correct but contents are not
        except ValueError:
            print("Invalid value in a field")
        #When contents of file are manipulated
        except NameError as ne:
            print("Variable cannot be found", ne)
        #Any other fault
        except:
            print("Unknown error happened")