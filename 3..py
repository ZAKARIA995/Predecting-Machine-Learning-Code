import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = 'Immunotherapy1.xlsx'

veriseti = pd.read_excel(PATH)


X = veriseti.iloc[:,:-1].values
Y = veriseti.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25 , random_state = 0)







from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test =scaler.fit_transform(y_test.reshape(-1,1))



from sklearn.neighbors import KNeighborsClassifier
siniflandirici = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
siniflandirici.fit(X_train, y_train)
y_pred = siniflandirici.predict(X_test)




from sklearn.metrics import confusion_matrix,classification_report
hm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))



_cinsiyet = 2
_yas = 26
_sure=11
_adet = 6
_tip = 1
_alan=50
_sertlesmeCapi=9

girdiler = np.array([_cinsiyet,_yas , _sure , _adet , _tip , _alan , _sertlesmeCapi])
girdiler = scaler.fit_transform(girdiler.reshape(1,-1))
y_pred2 = siniflandirici.predict(np.reshape(girdiler,(1,-1)))



hatalarListesi = []
for i in range(1,31):
    
    siniflandirici = KNeighborsClassifier(n_neighbors = i )
    siniflandirici.fit(X_train, y_train)
    pred_i = siniflandirici.predict(X_test)
    hatalarListesi.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,4))
plt.plot(range(1,31),hatalarListesi,'m_')   
    