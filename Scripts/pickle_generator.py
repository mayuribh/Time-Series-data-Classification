import pickle as pk
import os
import sys
import csv
import numpy as np
import pandas as pd

file_list = []
actual_csv = []
yes_list_from_csv = []
no_list_from_csv = []
up_list_from_csv = []
down_list_from_csv = []
next_list_from_csv = []
back_list_from_csv = []
normal_list_from_csv = []
train_set = []
test_set = []
val_set = []

def loader():
    global file_list

    os.path.dirname(sys.argv[0])
    print("Location is", os.path.dirname(sys.argv[0]) + "/Model Data/preprocessed learning data/usable")
    path=os.path.dirname(sys.argv[0]) + "/Model Data/preprocessed learning data/usable"
    #file_list = next(os.walk(os.path.dirname(sys.argv[0]) + "/Model Data/preprocessed learning data/usable"))[2]
    file_list = [os.path.join(path, f) for f in os.listdir(path)]
    file_list.sort()

    print("\n Data List loaded", file_list)


def file_loader():
    global yes_list_from_csv
    global no_list_from_csv
    global up_list_from_csv
    global down_list_from_csv
    global next_list_from_csv
    global back_list_from_csv
    global normal_list_from_csv
    global actual_csv
    global processing_list

    print("\nloading path is", os.path.dirname(sys.argv[0]) + "/Model Data/preprocessed learning data/usable/" + actual_csv)
    csv.register_dialect('myDialect1', delimiter=';')

    with open(actual_csv, 'r') as f:
        reader = csv.reader(f, dialect='myDialect1')

        n = -1
        print(actual_csv)
        for row in reader:
            n = n + 1
            print(actual_csv[90:93])
            if actual_csv[90:93] == "yes":
                yes_list_from_csv.append(row)
            if actual_csv[90:93] == "no_":
                no_list_from_csv.append(row)
            if actual_csv[90:93] == "up_":
                up_list_from_csv.append(row)
            if actual_csv[90:93] == "dow":
                down_list_from_csv.append(row)
            if actual_csv[90:93] == "nex":
                next_list_from_csv.append(row)
            if actual_csv[90:93] == "bac":
                back_list_from_csv.append(row)
           # if actual_csv[100:103] == "nor":
            #    normal_list_from_csv.append(row)


    print("\n File", actual_csv, "loaded")


def randomizer():
    global yes_list_from_csv
    global no_list_from_csv
    global up_list_from_csv
    global down_list_from_csv
    global next_list_from_csv
    global back_list_from_csv
    global normal_list_from_csv

    global train_set
    global test_set
    global val_set

    print(len(yes_list_from_csv))
    print(len(no_list_from_csv))
    print(len(up_list_from_csv))
    print(len(down_list_from_csv))
    print(len(next_list_from_csv))
    print(len(back_list_from_csv))
   # print(len(normal_list_from_csv))

    del (yes_list_from_csv[650:])
    del (no_list_from_csv[650:])
    del (up_list_from_csv[650:])
    del (down_list_from_csv[650:])
    del (next_list_from_csv[650:])
    del (back_list_from_csv[650:])
    #del (normal_list_from_csv[650:])

   # for i in range(len(normal_list_from_csv)):
    #    if len(normal_list_from_csv[i]) != 110:
    #       print(i)
     #       print(len(normal_list_from_csv[i]))

    train_set_number = 468
    test_set_number =130
    val_set_number = 52

    for i in range(train_set_number):
        train_set.append(yes_list_from_csv[i])
        train_set.append(no_list_from_csv[i])
        train_set.append(up_list_from_csv[i])
        train_set.append(down_list_from_csv[i])
        train_set.append(next_list_from_csv[i])
        train_set.append(back_list_from_csv[i])
       # train_set.append(normal_list_from_csv[i])

    for i in range(test_set_number):
        test_set.append(yes_list_from_csv[i + train_set_number])
        test_set.append(no_list_from_csv[i + train_set_number])
        test_set.append(up_list_from_csv[i + train_set_number])
        test_set.append(down_list_from_csv[i + train_set_number])
        test_set.append(next_list_from_csv[i + train_set_number])
        test_set.append(back_list_from_csv[i + train_set_number])
        #test_set.append(normal_list_from_csv[i + train_set_number])

    for i in range(val_set_number):
        val_set.append(yes_list_from_csv[i + train_set_number + test_set_number])
        val_set.append(no_list_from_csv[i + train_set_number + test_set_number])
        val_set.append(up_list_from_csv[i + train_set_number + test_set_number])
        val_set.append(down_list_from_csv[i + train_set_number + test_set_number])
        val_set.append(next_list_from_csv[i + train_set_number + test_set_number])
        val_set.append(back_list_from_csv[i + train_set_number + test_set_number])
        #val_set.append(normal_list_from_csv[i + train_set_number + test_set_number])

    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    np.random.shuffle(val_set)


def pickle_dumper():
    global train_set
    global test_set
    global val_set

    left_eye_x_list = []
    left_eye_y_list = []
    right_eye_x_list = []
    right_eye_y_list = []
    mean_x_list = []
    mean_y_list = []
    label_list = []
    wholelist=[]

    characters = ['[', ']']

    for i in range(len(train_set)):
        #transferlist1 = []
        transferlist2 = []
        transferlist3 = []
        transferlist4 = []
        transferlist5 = []
        transferlist6 = []
        transferlist7 = None
        transferlist=[]

   
        for x in range(len(train_set[i])):

            train_set[i][x] = train_set[i][x].split(', ')
            for v in range(len(train_set[i][x])):
                for character in characters:
                    train_set[i][x][v] = train_set[i][x][v].replace(character, "")

            transferlist1=[str(train_set[i][x][0]),str(train_set[i][x][1]),str(train_set[i][x][2]),str(train_set[i][x][3]),str(train_set[i][x][4]),str(train_set[i][x][5])]
            transferlist.append((transferlist1))

            """
            transferlist1.append(float(train_set[i][x][0]))
            transferlist2.append(float(train_set[i][x][1]))
            transferlist3.append(float(train_set[i][x][2]))
            transferlist4.append(float(train_set[i][x][3]))
            transferlist5.append(float(train_set[i][x][4]))
            transferlist6.append(float(train_set[i][x][5]))
            """
         #   if int(train_set[i][x][6][0]) == 0:
          #      transferlist7 = "normal"
            if int(train_set[i][x][6][0]) == 1:
                transferlist7 = "yes"
            if int(train_set[i][x][6][0]) == 2:
                transferlist7 = "no"
            if int(train_set[i][x][6][0]) == 3:
                transferlist7 = "up"
            if int(train_set[i][x][6][0]) == 4:
                transferlist7 = "down"
            if int(train_set[i][x][6][0]) == 5:
                transferlist7 = "next"
            if int(train_set[i][x][6][0]) == 6:
                transferlist7 = "back"

        wholelist.append(transferlist)
        label_list.append(transferlist7)

        """
        left_eye_x_list.append(transferlist1)
        left_eye_y_list.append(transferlist2)
        right_eye_x_list.append(transferlist3)
        right_eye_y_list.append(transferlist4)
        mean_x_list.append(transferlist5)
        mean_y_list.append(transferlist6)
        label_list.append(transferlist7)
        """
    print(len(wholelist))
    data_train_set = {'sequence':wholelist,
                     'label': label_list
                 }

    filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/train_half.csv"

    with open(filename,mode='a', newline='') as f:
            testwriter = csv.writer(f, delimiter=';')
            for i in range(len(wholelist)):
                  row=[(wholelist[i]),label_list[i]]
                  testwriter.writerow(row)
    """        
    df = pd.DataFrame(data = data_train_set)
    filename1 = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/train_set.csv"
                testwriter.writerow(testing[0][i])#file1 = open(filename1, 'wb')
    df.to_csv(filename1)
    
    filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/train_set.pkl"
    file = open(filename, 'wb')
    pk.dump(df, file)
    """

    # 2scnd list

    left_eye_x_list = []
    left_eye_y_list = []
    right_eye_x_list = []
    right_eye_y_list = []
    mean_x_list = []
    mean_y_list = []
    label_list = []
    wholelist=[]

    characters = ['[', ']']

    for i in range(len(test_set)):
        transferlist1 = []
        transferlist2 = []
        transferlist3 = []
        transferlist4 = []
        transferlist5 = []
        transferlist6 = []
        transferlist7 = None
        transferlist=[]

        for x in range(len(test_set[i])):

            test_set[i][x] = test_set[i][x].split(', ')
            for v in range(len(test_set[i][x])):
                for character in characters:
                    test_set[i][x][v] = test_set[i][x][v].replace(character, "")

            transferlist1=[float(test_set[i][x][0]),float(test_set[i][x][1]),float(test_set[i][x][2]),float(test_set[i][x][3]),float(test_set[i][x][4]),float(test_set[i][x][5])]
            transferlist.append(transferlist1)
            """
            transferlist1.append(float(test_set[i][x][0]))
            transferlist2.append(float(test_set[i][x][1]))
            transferlist3.append(float(test_set[i][x][2]))
            transferlist4.append(float(test_set[i][x][3]))
            transferlist5.append(float(test_set[i][x][4]))
            transferlist6.append(float(test_set[i][x][5]))
            """
           # if int(test_set[i][x][6][0]) == 0:
            #    transferlist7 = "normal"
            if int(test_set[i][x][6][0]) == 1:
                transferlist7 = "yes"
            if int(test_set[i][x][6][0]) == 2:
                transferlist7 = "no"
            if int(test_set[i][x][6][0]) == 3:
                transferlist7 = "up"
            if int(test_set[i][x][6][0]) == 4:
                transferlist7 = "down"
            if int(test_set[i][x][6][0]) == 5:
                transferlist7 = "next"
            if int(test_set[i][x][6][0]) == 6:
                transferlist7 = "back"

        wholelist.append(transferlist)
        """
        left_eye_x_list.append(transferlist1)
        left_eye_y_list.append(transferlist2)
        right_eye_x_list.append(transferlist3)
        right_eye_y_list.append(transferlist4)
        mean_x_list.append(transferlist5)
        mean_y_list.append(transferlist6)
        """
        label_list.append(transferlist7)

    data_test_set = {'sequence': wholelist,

                      'label': label_list
                      }
    filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/test_half.csv"
    with open(filename,mode='a', newline='') as f:
             testwriter = csv.writer(f, delimiter=';')
             for i in range(len(wholelist)):
                   row=[wholelist[i],label_list[i]]
                   testwriter.writerow(row)
    """
    filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/test_set.pkl"
    file = open(filename, 'wb')
    pk.dump(df, file)
    """



    # 3scnd list

    left_eye_x_list = []
    left_eye_y_list = []
    right_eye_x_list = []
    right_eye_y_list = []
    mean_x_list = []
    mean_y_list = []
    label_list = []
    wholelist=[]

    characters = ['[', ']']

    for i in range(len(val_set)):
        transferlist1 = []
        transferlist2 = []
        transferlist3 = []
        transferlist4 = []
        transferlist5 = []
        transferlist6 = []
        transferlist7 = None
        transferlist=[]

        for x in range(len(val_set[i])):

            val_set[i][x] = val_set[i][x].split(', ')
            for v in range(len(val_set[i][x])):
                for character in characters:
                    val_set[i][x][v] = val_set[i][x][v].replace(character, "")

            transferlist1=[float(val_set[i][x][0]),float(val_set[i][x][1]),float(val_set[i][x][2]),float(val_set[i][x][3]),float(val_set[i][x][4]),float(val_set[i][x][5])]
            transferlist.append(transferlist1)
            """
            transferlist2.append(float(val_set[i][x][1]))
            transferlist3.append(float(val_set[i][x][2]))
            transferlist4.append(float(val_set[i][x][3]))
            transferlist5.append(float(val_set[i][x][4]))
            transferlist6.append(float(val_set[i][x][5]))
            """
           # if int(val_set[i][x][6][0]) == 0:
            #    transferlist7 = "normal"
            if int(val_set[i][x][6][0]) == 1:
                transferlist7 = "yes"
            if int(val_set[i][x][6][0]) == 2:
                transferlist7 = "no"
            if int(val_set[i][x][6][0]) == 3:
                transferlist7 = "up"
            if int(val_set[i][x][6][0]) == 4:
                transferlist7 = "down"
            if int(val_set[i][x][6][0]) == 5:
                transferlist7 = "next"
            if int(val_set[i][x][6][0]) == 6:
                transferlist7 = "back"
        wholelist.append(transferlist)
        """
        left_eye_x_list.append(transferlist1)
        left_eye_y_list.append(transferlist2)
        right_eye_x_list.append(transferlist3)
        right_eye_y_list.append(transferlist4)
        mean_x_list.append(transferlist5)
        mean_y_list.append(transferlist6)
        """
        label_list.append(transferlist7)

    data_val_set = {'sequence':wholelist,
                      'label': label_list
                      }

    df = pd.DataFrame(data=data_val_set)
    filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/val_half.csv"
    with open(filename,mode='a', newline='') as f:
              testwriter = csv.writer(f, delimiter=';')                           #filename = os.path.dirname(sys.argv[0]) + "/Model Data/pickle/val_set.csv"
              for i in range(len(wholelist)):                                     #df.to_csv(filename)
                    row=[wholelist[i],label_list[i]]
                    testwriter.writerow(row)                                      #file = open(filename, 'wb')
    #pk.dump(df, file)



def test():
    prof_set = pd.read_pickle(os.path.dirname(sys.argv[0]) + "/Model Data/pickle/Archiv/train_set.pkl")
    train_set = pd.read_pickle(os.path.dirname(sys.argv[0]) + "/Model Data/pickle/train_set.pkl")
    test = train_set['label']
    test_prof = prof_set['label']



    temp1 = ['normal', 'yes', 'no', 'up', 'down', 'next', 'back']
    temp = ['Normal', 'gesture1', 'gesture2', 'gesture3', 'gesture4', 'gesture5', 'gesture6']

    train_label1 = []
    train_label2 = []

    for i in test_prof:
            idx = temp.index(i)
            train_label2.append(idx)

    for i in test:
            idx = temp1.index(i)
            train_label1.append(idx)


    a1 = np.asarray(train_set['left_eye_x'])
    b1 = np.asarray(train_set['left_eye_y'])
    a2 = np.asarray(prof_set['gaze_data_x'])
    b2 = np.asarray(prof_set['gaze_data_y'])

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(0, a1.shape[0]):
        row1 = a1[i]
        row2 = b1[i]
        x1.append(row1)
        y1.append(row2)

    for i in range(0, a2.shape[0]):
        row1 = a2[i]
        row2 = b2[i]
        x2.append(row1)
        y2.append(row2)

def logic():

    global file_list
    global actual_csv

    loader()

    for i in range(len(file_list)):
        actual_csv = file_list[i]
        file_loader()

    randomizer()
    pickle_dumper()






logic()
