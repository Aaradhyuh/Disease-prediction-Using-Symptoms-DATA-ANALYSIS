from tkinter import *
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

pd.set_option('future.no_silent_downcasting', False)

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer disease','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)
### yahan par simply l2 ME 0s fill kar rhe hai same lenght of l1 prediction ke liye  10-> Back pain
# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)
## yahan par disease ko numeric value se replace kar rhe hai hum df me since models ke liye ez hoga numeric values se deal karna since unhe strings se deal karna nhi aata

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)


# print(y)
## yahan par humne X me saare symptoms bhare hai and as a input act karenge
##y will be our final/target
## x is a data frame and y is 1 d numpy array taaki models read kar ske



# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

def DecisionTree():
    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    # Accuracy calculation
    from sklearn.metrics import accuracy_score
    y_pred = clf3.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Reset symptoms list
    global l2
    l2 = [0] * len(l1)

    # Collect symptoms
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Validate symptoms
    if not all(psymptoms):
        t1.delete("1.0", END)
        t1.insert(END, "Please select all symptoms.")
        return

    # Update l2 with selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Prepare input and predict
    inputtest = pd.DataFrame([l2], columns=l1)
    #l2 ke checked symptoms and rest columns of l1
    predict = clf3.predict(inputtest)
    predicted = predict[0]

    # Display result
    if predicted in range(len(disease)):
        t1.delete("1.0", END)
        t1.insert(END, disease[predicted])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Prediction not found.")

def randomforest():
    from sklearn.ensemble import RandomForestClassifier

    # Train the Random Forest model
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    y_pred = clf4.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Reset symptoms list
    global l2
    l2 = [0] * len(l1)

    # Collect symptoms
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Validate symptoms
    if not all(psymptoms):
        t2.delete("1.0", END)
        t2.insert(END, "Please select all symptoms.")
        return

    # Update l2 with selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Prepare input and predict
    inputtest = pd.DataFrame([l2], columns=l1)
    predict = clf4.predict(inputtest)
    predicted = predict[0]

    # Display result
    if predicted in range(len(disease)):
        t2.delete("1.0", END)
        t2.insert(END, disease[predicted])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Prediction not found.")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB

    # Train the Naive Bayes model
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    y_pred = gnb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Reset symptoms list
    global l2
    l2 = [0] * len(l1)

    # Collect symptoms
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Validate symptoms
    if not all(psymptoms):
        t3.delete("1.0", END)
        t3.insert(END, "Please select all symptoms.")
        return

    # Update l2 with selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Prepare input and predict
    inputtest = pd.DataFrame([l2], columns=l1)
    predict = gnb.predict(inputtest)
    predicted = predict[0]

    # Display result
    if predicted in range(len(disease)):
        t3.delete("1.0", END)
        t3.insert(END, disease[predicted])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Prediction not found.")
def KNN():
    from sklearn.neighbors import KNeighborsClassifier

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust 'n_neighbors' as needed
    knn.fit(X, np.ravel(y))

    # Calculate accuracy
    y_pred = knn.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Reset symptoms list
    global l2
    l2 = [0] * len(l1)

    # Collect symptoms
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Validate symptoms
    if not all(psymptoms):
        t4.delete("1.0", END)
        t4.insert(END, "Please select all symptoms.")
        return

    # Update l2 with selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Prepare input and predict
    inputtest = pd.DataFrame([l2], columns=l1)
    predict = knn.predict(inputtest)
    predicted = predict[0]

    # Display result
    if predicted in range(len(disease)):
        t4.delete("1.0", END)
        t4.insert(END, disease[predicted])
    else:
        t4.delete("1.0", END)
        t4.insert(END, "Prediction not found.")

def LogisticRegressionModel():
    from sklearn.linear_model import LogisticRegression

    # Train the Logistic Regression model
    logreg = LogisticRegression(max_iter=500)  # Increase max_iter for convergence
    logreg.fit(X, np.ravel(y))

    # Calculate accuracy
    y_pred = logreg.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Reset symptoms list
    global l2
    l2 = [0] * len(l1)

    # Collect symptoms
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    # Validate symptoms
    if not all(psymptoms):
        t5.delete("1.0", END)
        t5.insert(END, "Please select all symptoms.")
        return

    # Update l2 with selected symptoms
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    # Prepare input and predict
    inputtest = pd.DataFrame([l2], columns=l1)
    predict = logreg.predict(inputtest)
    predicted = predict[0]

    # Display result
    if predicted in range(len(disease)):
        t5.delete("1.0", END)
        t5.insert(END, disease[predicted])
    else:
        t5.delete("1.0", END)
        t5.insert(END, "Prediction not found.")

# gui_lesgoo

root = Tk()
root.geometry('1600x1600')  # Adjust size of the window
root.configure(background='#DDEFF5')

# entry variables
Symptom1 = StringVar()
Symptom1.set("Select Symptom 1")
Symptom2 = StringVar()
Symptom2.set("Select Symptom 2")
Symptom3 = StringVar()
Symptom3.set("Select Symptom 3")
Symptom4 = StringVar()
Symptom4.set("Select Symptom 4")
Symptom5 = StringVar()
Symptom5.set("Select Symptom 5")

#Name = StringVar()



w2 = Label(root, justify=CENTER, text="Disease Prediction Using Big DATA", fg="#004466", bg="#DDEFF5")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=300)


S1Lb = Label(root, text="Symptom 1", fg="#333333", bg="#BFD9E8")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="#333333", bg="#BFD9E8")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="#333333", bg="#BFD9E8")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="#333333", bg="#BFD9E8")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="#333333", bg="#BFD9E8")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="DecisionTree", fg="white", bg="#7D9EA6")
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="RandomForest", fg="white", bg="#7D9EA6")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="white", bg="#7D9EA6")
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

KNN2= Label(root, text="KNN (Inaccurate)", fg="white", bg="#7D9EA6")
KNN2.grid(row=21, column=0, pady=10, sticky=W)

LOG = Label(root, text="Logistic", fg="white", bg="#7D9EA6")
LOG.grid(row=23, column=0, pady=10, sticky=W)
# entries
OPTIONS = sorted(l1)



S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)


dst = Button(root, text="Predict (Decision Tree)", command=DecisionTree, bg="#9ACD9A", fg="#333333")
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Predict (Random Forest)", command=randomforest, bg="#9ACD9A", fg="#333333")
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="Predict (Naive Bayes)", command=NaiveBayes, bg="#9ACD9A", fg="#333333")
lr.grid(row=10, column=3,padx=10)

knn_btn = Button(root, text="Predict (KNN)", command=KNN, bg="#9ACD9A", fg="#333333")
knn_btn.grid(row=11, column=3, padx=10)

logreg_btn = Button(root, text="Predict (Logistic Regression)", command=LogisticRegressionModel, bg="#9ACD9A", fg="#333333")
logreg_btn.grid(row=12, column=3, padx=10)


#textfileds
t1 = Text(root, height=1, width=40,bg="#FFF5BA",fg="#333333")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="#FFF5BA",fg="#333333")
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="#FFF5BA",fg="#333333")
t3.grid(row=19, column=1 , padx=10)

t4 = Text(root, height=1, width=40, bg="#FFF5BA", fg="#333333")
t4.grid(row=21, column=1, padx=10)

t5 = Text(root, height=1, width=40, bg="#FFF5BA", fg="#333333")
t5.grid(row=23, column=1, padx=10)


root.mainloop()
