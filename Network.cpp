#include "Network.h"
/* Инициализация и заполнение всех свойства класса Network, кроме neurons_val*/
void Network::Init(data_Network data) {
	actFunc.set();
	srand(time(NULL)); // инициализиция рандомайзера
	L = data.L; // запоминаем кол-во слоев в нейронной сети
	size = new int[L]; // выделение памяти под кол-во нейронов в каждом слое
	for (int i = 0; i < L; i++) {
		size[i] = data.size[i]; // заполнение значениями кол-во нейронов в каждом слое
	}
	weights = new Matrix[L - 1]; // выделение памяти под матрицу весов
	bios = new double* [L - 1]; //  выделение памяти под веса смещения
	for (int i = 0; i < L - 1; i++) { // заполнение значениями весов и весов смещения
		weights[i].Init(size[i + 1], size[i]);
		bios[i] = new double[size[i + 1]];
		weights[i].Rand();
		for (int j = 0; j < size[i + 1]; j++) {
			bios[i][j] = ((rand() % 50)) * 0.06 / (size[i] + 15);
		}
	}
	neurons_val = new double* [L]; // выделение памяти под
	neurons_err = new double* [L]; // выделение памяти под
	for (int i = 0; i < L; i++) {
		neurons_val[i] = new double[size[i]]; // заполнение значениями нейронов
		neurons_err[i] = new double[size[i]]; // заполнение значениями ошибок на нейронах
	}
	neurons_bios_val = new double[L - 1]; // выделение памяти под значения нейронов смещения
	for (int i = 0; i < L - 1; i++) { 
		neurons_bios_val[i] = 1; // заполнение значениями нейронов смещения
	}
}

/* Вывод на экран кол-ва слоев сети и массива кол-ва нейронов в каждом слое*/
void Network::PrintConfig() {
	cout << "**********\n";
	cout << "Network has " << L << " layers\nSIZE[]: ";
	for (int i = 0; i < L; i++) {
		cout << size[i] << " ";
	}
	cout << "\n**********\n\n";
}

/* Заполнение массива neurons_val
	values - данные (изображения MNIST) */
void Network::SetInput(double* values) {
	for (int i = 0; i < size[0]; i++) {
		neurons_val[0][i] = values[i];
	}
}

// expect - "правильная" цифра
void Network::BackPropogation(double expect) {
	// расчет дельты для выходных нейронов
	for (int i = 0; i < size[L - 1]; i++) {
		if (i != int(expect)) {
			neurons_err[L - 1][i] = -neurons_val[L - 1][i] * actFunc.useDer(neurons_val[L - 1][i]);
		}
		else {
			neurons_err[L - 1][i] = (1.0 - neurons_val[L - 1][i]) * actFunc.useDer(neurons_val[L - 1][i]);
		}
	}
	// расчет дельты для скрытых нейронов
	for (int k = L - 2; k > 0; k--) {
		Matrix::Multi_T(weights[k], neurons_err[k + 1], size[k + 1], neurons_err[k]);
		for (int j = 0; j < size[k]; j++) {
			neurons_err[k][j] *= actFunc.useDer(neurons_val[k][j]);
		}
	}
}

// Обновление весов
void Network::WeightsUpdater(double lr) {
	for (int i = 0; i < L - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			for (int k = 0; k < size[i]; ++k) {
				weights[i](j, k) += neurons_val[i][k] * neurons_err[i + 1][j] * lr;
			}
		}
	}
	for (int i = 0; i < L - 1; ++i) {
		for (int k = 0; k < size[i + 1]; k++) {
			bios[i][k] += neurons_err[i + 1][k] * lr;
		}
	}
}

// Сохранение весов в файл
void Network::SaveWeights() {
	ofstream fout;
	fout.open("Weights.txt");
	if (!fout.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < L - 1; ++i) {
		fout << weights[i] << " ";
	}
	for (int i = 0; i < L - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fout << bios[i][j] << " ";
		}
	}
	cout << "Weights saved \n";
	fout.close();
}

// Чтение весов из файла
void Network::ReadWeights() {
	ifstream fin;
	fin.open("Weights.txt");
	if (!fin.is_open()) {
		cout << "Error reading the file";
		system("pause");
	}
	for (int i = 0; i < L - 1; ++i) {
		fin >> weights[i];
	}
	for (int i = 0; i < L - 1; ++i) {
		for (int j = 0; j < size[i + 1]; ++j) {
			fin >> bios[i][j];
		}
	}
	cout << "Weights readed \n";
	fin.close();
}

double Network::ForwardFeed() {
	for (int k = 1; k < L; ++k) {
		// матрицу весов умножаем на вектор-столбец значений нейронов, также передаем размер и что хотим получить
		Matrix::Multi(weights[k - 1], neurons_val[k - 1], size[k - 1], neurons_val[k]);
		// суммирование полученного вектора с биосом
		Matrix::SumVector(neurons_val[k], bios[k - 1], size[k]);
		// используем фукнцию активации
		actFunc.use(neurons_val[k], size[k]);
	}
	int pred = SearchMaxIndex(neurons_val[L - 1]);
	return pred;
}

/* Получение ответа от нейросети 
   value - вектор значений */
int Network::SearchMaxIndex(double* value) {
	double max = value[0];
	int prediction = 0; // индекс максимального элемента
	double temp;
	for (int j = 1; j < size[L - 1]; j++) {
		temp = value[j];
		if (temp > max) {
			prediction = j;
			max = temp;
		}
	}
	return prediction;
}

/* Вывод на экран индекса и значения нейрона на слое L */
void Network::PrintValues(int L) {
	for (int j = 0; j < size[L]; j++) {
		cout << j << " " << neurons_val[L][j] << endl;
	}
}