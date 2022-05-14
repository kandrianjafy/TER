from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os


def threshold(img_dir, thresh):

    os.makedirs(img_dir+'thresholded/', exist_ok=True)

    img_list = []
    for file in os.listdir(img_dir):
        if file.endswith(".jpg"):
            img_list.append(img_dir+file)

    for salade in tqdm(img_list, desc="Thresholding"):
        img = cv.imread(salade, cv.IMREAD_GRAYSCALE)
        fname = salade.split('/')[-1]

        thresholded_img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]
        thresholded_img = cv.dilate(thresholded_img, None, iterations=2)
        thresholded_img = cv.erode(thresholded_img, None, iterations=14)

        # save thresholded image
        cv.imwrite(img_dir+'/thresholded/'+fname, thresholded_img)


def fitellipse(img_dir):

    os.makedirs(img_dir+'fitellipse/', exist_ok=True)

    excentricite = []

    thresholded_dir = img_dir + '/thresholded/'
    img_list = []
    for file in os.listdir(thresholded_dir):
        if file.endswith(".jpg"):
            img_list.append(thresholded_dir+file)

    for salade in tqdm(img_list, desc="Fitting ellipse"):
        img = cv.imread(salade)
        fname = salade.split('/')[-1]
        # convert image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # find contours
        contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # find the biggest contour
        cnt = max(contours, key=cv.contourArea)
        # fit ellipse to the biggest contour
        ellipse = cv.fitEllipse(cnt)
        (x, y), (a, b), theta = ellipse
        excentricite.append((1-a**2/b**2)**0.5)
        # draw the ellipse
        cv.ellipse(img, ellipse, (0, 255, 0), 15)
        # save the image
        cv.imwrite(img_dir+'/fitellipse/'+fname, img)

    return excentricite


def optimal_threshold(faciee_dir, pas_faciee_dir):

    def excentricite_calcul(img_dir, thresh):

        img_list = []
        thresh_img = []

        for file in os.listdir(img_dir):
            if file.endswith(".jpg"):
                img_list.append(img_dir+file)

        for salade in img_list:
            img = cv.imread(salade,cv.IMREAD_GRAYSCALE)

            thresholded_img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]
            thresholded_img = cv.dilate(thresholded_img, None, iterations=2)
            thresholded_img = cv.erode(thresholded_img, None, iterations=18)
            thresh_img.append(thresholded_img)

        excentricite = []

        for img in thresh_img:
            try :
                #find contours
                contours, _= cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                #find the biggest contour
                cnt = max(contours, key=cv.contourArea)
                #fit ellipse to the biggest contour
                ellipse = cv.fitEllipse(cnt)
                (x, y), (a, b), theta = ellipse
                excentricite.append((1-a**2/b**2)**0.5)
                #draw the ellipse
                cv.ellipse(img, ellipse, (0,255,0), 5)
            except:
                excentricite.append(0.5)
                continue

        return excentricite
    
    mean_ex_faciee = []
    mean_ex_pas_faciee = []

    for seuil in tqdm(range(0,256), desc="Computing mean excentricity"):
        mean_ex_faciee.append(np.mean(excentricite_calcul(faciee_dir, seuil)))
        mean_ex_pas_faciee.append(np.mean(excentricite_calcul(pas_faciee_dir, seuil)))
    
    index = 0
    max_diff=0
    for i in range(len(mean_ex_faciee)):
        diff = mean_ex_faciee[i]-mean_ex_pas_faciee[i]
        if diff > max_diff:
            max_diff = diff
            index = i
    
    return index


def roc_curve(y_test, y_actu):

    # compute y_pred for different threshold
    linspace = np.linspace(0, 1, 40)
    def y_pred_fct(y_test):
        liste = []
        for k in tqdm(linspace, desc="Computing ROC curve"):
            y_pred = [1 if i >= k else 0 for i in y_test]
            liste.append(y_pred)
        return liste
    
    y_pred_list = y_pred_fct(y_test)

    # confusion matrix computation for different y_pred
    TPR = [] # sensitivity, recall or true positive rate
    FPR = [] # false positive rate or 1 - specificity
    auc=0

    for i in tqdm(y_pred_list):
        global score

        y_pred = np.array(i)
        cnf_matrix = metrics.confusion_matrix(y_actu, y_pred)

        if metrics.roc_auc_score(y_actu, y_pred) >= auc:
            auc = metrics.roc_auc_score(y_actu, y_pred)
            score = linspace[y_pred_list.index(i)]

        TP = cnf_matrix[1, 1]
        TN = cnf_matrix[0, 0]
        FP = cnf_matrix[0, 1]
        FN = cnf_matrix[1, 0]

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        # compute true positive and false positive rates
        TPR.append(sensitivity)
        FPR.append(1 - specificity)


    # plot ROC curve and save it
    plt.xlim = [0, 1]
    plt.ylim = [0, 1]
    plt.plot([0.0, 1], [0.0, 1], 'r--', lw=2)
    plt.xlabel("FPR (1-Spécificité)")
    plt.ylabel("TPR (Sensibilité)")
    plt.plot(FPR, TPR)
    plt.savefig("ROC_curve.png")


    # calcul de l'AUC

    def auc(FPR, TPR):
        auc = 0
        n = len(FPR)
        for i in range(1,n):
            auc += (FPR[i-1]-FPR[i]) * ((TPR[i-1]+TPR[i])/2)
        return auc

    modele_auc = auc(FPR, TPR)
    
    return modele_auc, score