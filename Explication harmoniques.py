import numpy as np
import matplotlib.pyplot as plt



def afficher_harmoniques(freq):
    fig, ax = plt.subplots()
    t = np.arange(0,2,0.01)
    fondamentale =  np.sin(2 * np.pi * freq * t) + 40
    h2 =  1/2 * np.sin(2 * np.pi * 2 * freq * t) + 30
    h3 =  1/3 * np.sin(2 * np.pi * 3 * freq * t) + 20
    h4 =  1/4 * np.sin(2 * np.pi * 4 * freq * t) + 10
    somme = fondamentale + h2 + h3 + h4 - 100
    ax.plot(fondamentale)
    ax.text(100, 35, '+', fontsize=30, ha='center', va='center')
    ax.plot(h2)
    ax.text(100, 25, '+', fontsize=30, ha='center', va='center')
    ax.plot(h3)
    ax.text(100, 15, '+', fontsize=30, ha='center', va='center')
    ax.plot(h4)
    ax.text(100, 5, '=', fontsize=30, ha='center', va='center')
    ax.plot(somme)
    ax.set(xlabel="Temps (en s)", ylabel="Amplitude", title="Explication de la composition d'un son produit par une corde")
    plt.show()

afficher_harmoniques(5)
    
    
    