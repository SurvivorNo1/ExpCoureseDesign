import joblib
from sklearn.svm import SVC
import os

def train(outpath,trainloader):
    datas = trainloader.dataset.data.reshape(-1, 28*28)
    labels = trainloader.dataset.labels.argmax(axis=1)
    model = SVC(kernel='rbf', C=1.0, random_state=0,decision_function_shape='ovr')
    model.fit(datas, labels)
    joblib.dump(model, os.path.join(outpath,"svm.pkl"))
    return model

def test(model,testloader,name=""):
    test_datas = testloader.dataset.data.reshape(-1, 28*28)
    test_labels = testloader.dataset.labels.argmax(axis=1)
    y_hat = model.predict(test_datas)
    return y_hat, test_labels