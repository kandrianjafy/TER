from function import threshold, fitellipse, optimal_threshold, roc_curve
from tkinter import *
from tkinter import filedialog, ttk
from configparser import ConfigParser
from tqdm import tqdm
import os, csv

config = ConfigParser()
config.read('config.ini')

# default thresholding value set to 241
thresh = int(config['DEFAULT']['OptimalThreshold'])
score = float(config['DEFAULT']['Score'])

# creates a Tk() object
root = Tk()

# sets the geometry of main
# root window
root.resizable(False, False)
root.columnconfigure(1, weight=1)

# reset and dir selection buttons
def reset():
    global fasciee_dir
    global pasfaciee_dir
    global testdata_dir
    global y_test

    fasciee_dir, pasfaciee_dir, testdata_dir, y_test = None, None, None, None
    fasciee_button.config(state=NORMAL)
    pasfasciee_button.config(state=NORMAL)
    testdata_button.config(state=NORMAL)
    reset_button.config(state=DISABLED)
    print('fasciated dir set to None')
    print('not fasciated dir set to None')
    print('test data dir set to None')

def fasciee_folder():
    global fasciee_dir
    global class1_count
    fasciee_dir = filedialog.askdirectory() + '/'
    if fasciee_dir != '/':
        class1_count = len([name for name in os.listdir(fasciee_dir) if os.path.isfile(os.path.join(fasciee_dir,name))])
        print('fasciated dir set to ', fasciee_dir, 'containing ', class1_count, ' files')
        fasciee_button.config(state=DISABLED)
        reset_button.config(state=NORMAL)

def pasfasciee_folder():
    global pasfaciee_dir
    global class0_count
    pasfaciee_dir = filedialog.askdirectory() + '/'
    if pasfaciee_dir != '/':
        class0_count = len([name for name in os.listdir(pasfaciee_dir) if os.path.isfile(os.path.join(pasfaciee_dir,name))])
        print('not fasciated dir set to ', pasfaciee_dir, 'containing ', class0_count, ' files')
        pasfasciee_button.config(state=DISABLED)
        reset_button.config(state=NORMAL)

def testdata_folder():
    global testdata_dir
    testdata_dir = filedialog.askdirectory() + '/'
    if testdata_dir != '/':
        data_count = len([name for name in os.listdir(testdata_dir) if os.path.isfile(os.path.join(testdata_dir,name))])
        print('test data dir set to ', testdata_dir, 'containing ', data_count, ' files')
        testdata_button.config(state=DISABLED)
        reset_button.config(state=NORMAL)


# option part
def threshold_and_fitellipse():
    global fasciee_dir
    global pasfaciee_dir
    global y_test

    try: fasciee_dir
    except NameError: fasciee_dir = None
    try: pasfaciee_dir
    except NameError: pasfaciee_dir = None

    fasciee_dir = None if fasciee_dir == '/' else fasciee_dir
    pasfaciee_dir = None if pasfaciee_dir == '/' else pasfaciee_dir

    directories = [dir for dir in [fasciee_dir, pasfaciee_dir] if dir]
    y_test = []
    if directories==[]:
        print('No directory selected')
    else:
        for directory in directories:
            threshold(directory, thresh)
        print('Threshold done')

        for directory in directories:
            y_test + fitellipse(directory)
        print('Fitellipse done')

def compute_optimal_threshold():
    global fasciee_dir
    global pasfaciee_dir
    global thresh

    try: fasciee_dir
    except NameError: fasciee_dir = None
    try: pasfaciee_dir
    except NameError: pasfaciee_dir = None

    fasciee_dir = None if fasciee_dir == '/' else fasciee_dir
    pasfaciee_dir = None if pasfaciee_dir == '/' else pasfaciee_dir

    if None not in {fasciee_dir, pasfaciee_dir}:
        thresh = optimal_threshold(fasciee_dir, pasfaciee_dir)
        print('Optimal threshold: ', thresh)
    else:
        print('Make sure you select both train directories')

def compute_curveroc():
    global fasciee_dir
    global pasfaciee_dir
    global y_test
    global score

    try: fasciee_dir
    except NameError: fasciee_dir = None
    try: pasfaciee_dir
    except NameError: pasfaciee_dir = None
    try: y_test
    except NameError: y_test = None
    
    if None not in {fasciee_dir, pasfaciee_dir}:
        directories = [dir for dir in [fasciee_dir, pasfaciee_dir] if dir]
        if not y_test and directories != []:
            y_test=[]
            for directory in directories:
                threshold(directory, thresh)

            for directory in directories:
                y_test += fitellipse(directory)

        y_actu = [1] * class1_count + [0] * class0_count
        auc, score = roc_curve(y_test, y_actu)
        print('AUC: ', auc)
    else:
        print('Make sure you select both train directories')

def predict():
    global testdata_dir
    global score

    try: testdata_dir
    except NameError: testdata_dir = None

    testdata_dir = None if testdata_dir == '/' else testdata_dir

    if not testdata_dir:
        print('Make sure test data directory is selected')
    else:
        threshold(testdata_dir, thresh)
        excentricity = fitellipse(testdata_dir)

        img_list = []
        for file in os.listdir(testdata_dir):
            if file.endswith(".jpg"):
                img_list.append(file)
        
        nbr = len(img_list)
        with open('result.csv', 'a', newline='') as f:
            f.truncate(0)
            writer = csv.writer(f)
            for i in tqdm(range(nbr), desc="Creating result file"):
                writer.writerow([img_list[i], 1 if excentricity[i]>=score else 0])
        print('Prediction done')


# GUI part
sep = ttk.Separator(root,orient='horizontal')
sep.grid(row=1, column=1, sticky=EW)

traindata_label = Label(root, text="Load train set :")
traindata_label.grid(row=1, column=1)

fasciee_button = Button(text="fasciated directory", command=fasciee_folder)
fasciee_button.grid(row=2, column=1, sticky=EW, padx=50, pady=5)

pasfasciee_button = Button(text="not fasciated directory", command=pasfasciee_folder)
pasfasciee_button.grid(row=3, column=1, sticky=EW, padx=50, pady=5)

sep = ttk.Separator(root,orient='horizontal')
sep.grid(row=4, column=1, sticky=EW)

traindata_label = Label(root, text="Load test set :")
traindata_label.grid(row=4, column=1)

testdata_button = Button(text="test set directory", command=testdata_folder)
testdata_button.grid(row=5, column=1, sticky=EW, padx=50, pady=5)

sep = ttk.Separator(root,orient='horizontal')
sep.grid(row=6, column=1, sticky=EW)

traindata_label = Label(root, text="Options :")
traindata_label.grid(row=6, column=1)

threshold_button = Button(text="threshold & fitellipse", command=threshold_and_fitellipse)
threshold_button.grid(row=7, column=1, sticky=EW, padx=50, pady=5)

optimalthreshold_button = Button(text="find optimal threshold", command=compute_optimal_threshold)
optimalthreshold_button.grid(row=8, column=1, sticky=EW, padx=50, pady=5)

curveroc_button = Button(text="compute ROC curve", command=compute_curveroc)
curveroc_button.grid(row=9, column=1, sticky=EW, padx=50, pady=5)

predict_button = Button(text="predict", command=predict)
predict_button.grid(row=10, column=1, sticky=EW, padx=50, pady=5)


sep = ttk.Separator(root,orient='horizontal')
sep.grid(row=12, column=1, sticky=EW)


reset_button = Button(text="reset", command=reset)
reset_button.grid(row=13, column=1, pady=10)
reset_button.config(state=DISABLED)
mainloop()
