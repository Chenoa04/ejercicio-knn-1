import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# conjunto de entrenamiento
datos = {
    'Caracteristica1': [2, 4, 1, 2, 2, 2, 3, 3],
    'Caracteristica2': [0, 4, 1, 4, 2, 3, 4, 3],
    'Clase': [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(datos)
A_entrenamiento = df[['Caracteristica1', 'Caracteristica2']].values
B_entrenamiento = df['Clase'].values


clasificador = KNeighborsClassifier(n_neighbors=3, metric='manhattan')


clasificador.fit(A_entrenamiento, B_entrenamiento)

caso_a_clasificar = np.array([[2.5, 2.5]])


clase_predicha = clasificador.predict(caso_a_clasificar)

print(f"La clase predicha para el caso (2.5, 2.5) es: {clase_predicha[0]}")
