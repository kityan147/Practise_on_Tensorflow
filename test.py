import matplotlib.pyplot as plt
import numpy as np

def mergesort(a):
    if (len(a) == 1):
        return
    
    mid = int(len(a) / 2)
    L = a[:mid]
    R = a[mid:]

    mergesort(L)
    mergesort(R)

    i = 0
    j = 0
    k = 0
    while (i < len(L) and j < len(R)):
        if (L[i] < R[j]):
            a[k] = L[i]
            i = i + 1
        else:
            a[k] = R[j]
            j = j + 1
        k = k + 1
    
    while(i < len(L)):
        a[k] = L[i]
        i = i + 1
        k = k + 1
    
    while (j < len(R)):
        a[k] = R[j]
        j = j + 1
        k = k + 1
            
def main():
    a = [20, 14, 13, 7, 5, 3, 2,6, 8, 11, 12, 10]
    mergesort(a)
    for i in range(len(a)):
        print(a[i])
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
    plt.show()

if __name__ == "__main__":
    main()