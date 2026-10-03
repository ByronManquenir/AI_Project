"""Clasificador KNN sencillo para recomendar categorías de productos."""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


CARACTERISTICAS = ["precio", "popularidad", "duracion"]

# Cada fila contiene precio, popularidad y duración; los valores son ficticios.
PRODUCTOS = [
	[12, 25, 18], [18, 42, 30], [24, 35, 22], [29, 50, 40], [15, 30, 45],
	[22, 48, 15], [10, 20, 28], [27, 38, 35], [16, 52, 24], [20, 33, 42],
	[35, 88, 32], [42, 95, 45], [55, 82, 60], [62, 91, 38], [48, 78, 52],
	[39, 85, 25], [68, 97, 66], [51, 89, 44], [45, 80, 58], [59, 93, 30],
	[78, 45, 82], [92, 58, 95], [85, 35, 72], [73, 62, 88], [98, 50, 76],
	[81, 68, 93], [88, 40, 65], [76, 55, 98], [95, 30, 84], [83, 60, 78],
]

CATEGORIAS = (
	["Economico"] * 10
	+ ["Popular"] * 10
	+ ["Premium"] * 10
)


def entrenar_y_evaluar():
	"""Evalúa el modelo y lo devuelve reentrenado con todos los datos."""
	X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
		PRODUCTOS,
		CATEGORIAS,
		test_size=0.25,
		random_state=42,
		stratify=CATEGORIAS,
	)

	modelo = make_pipeline(
		StandardScaler(),
		KNeighborsClassifier(n_neighbors=3),
	)
	modelo.fit(X_entrenamiento, y_entrenamiento)
	precision = modelo.score(X_prueba, y_prueba)

	# Una vez medida la precisión, se aprovechan todos los ejemplos para predecir.
	modelo.fit(PRODUCTOS, CATEGORIAS)
	return modelo, precision


MODELO, PRECISION = entrenar_y_evaluar()


def recomendar_categoria(precio, popularidad, duracion):
	"""Predice la categoría de un producto a partir de sus tres características."""
	producto = [[precio, popularidad, duracion]]
	return MODELO.predict(producto)[0]


if __name__ == "__main__":
	print(f"Precisión en el conjunto de prueba: {PRECISION:.2%}")

	categoria = recomendar_categoria(precio=45, popularidad=90, duracion=50)
	print(f"Categoría recomendada para el producto nuevo: {categoria}")
