from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split

def loo(X_train, y_train, h_max: int) -> int:
    X = X_train.values
    Y = y_train.values
    max_percent = 0
    cur_percent = 0
    best_h = 0

    for h in range(1, h_max+1):
        correct = 0
        for i in range(len(X)):
            classes = [0, 0]
            for j in range (len(X)):
                if (i==j):
                    continue
                if (core(distance(X[i], X[j])/h)):
                    classes[Y[j][0]]+=1
            #if classes[0] + classes[1] == 0:
            #    continue
            if classes[0] >= classes[1]:
                calc = 0
            else:
                calc = 1
            if (calc == Y[i][0]):
                correct+=1
        cur_percent = correct/len(X) * 100
        print(f"Current percentage h = {h}: {cur_percent:.2f}%")
        if (cur_percent > max_percent):
            best_h = h
            max_percent = cur_percent
        else:
            break
    return best_h


def parzen(X_test, Y_test, h):
    correct = 0
    X = X_test.values
    Y = Y_test.values
    total = len(X)
    for i in range(len(X)):
        classes = [0, 0]
        for j in range (len(X)):
            if (i==j):
                continue
            if (core(distance(X[i], X[j])/h)):
                classes[Y[j][0]]+=1
        #if classes[0] + classes[1] == 0:
        #    continue
        if classes[0] >= classes[1]:
            calc = 0
        else:
            calc = 1
        if (calc == Y[i][0]):
            correct+=1
    return correct/total

def core(r) -> bool:
    return abs(r) <= 1 

def distance(x, y):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1]) ** 2)

def main():
    print("Вариант = 16")
    print("Метод парзеновского окна с фиксированным h")
    print("Ядро: П – прямоугольное K(x) = [r <= 1]")
    print("Файл номер 4")

    df = pd.read_csv('data4.csv')

    X = df[['MrotInHour', 'Salary']]
    Y = df[['Class']]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)

    h = loo(X_train, y_train, 10)
    print(f"Max percentage: {h}")
    result = parzen(X_test, y_test, h)
    print(f"The correctness of the choice = {result*100:.2f}%")

if __name__ == "__main__":
    main()