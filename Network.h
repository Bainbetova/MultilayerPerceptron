#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>

using namespace std;

// Для сравнения нейросетей разных конфигураций
struct data_Network
{
	int L; // количество слоев
	int* size; // количество нейронов в каждом слое
};

class Network
{
	int L; // количество слоев
	int* size; // количество нейронов в каждом слое
	ActivateFunction actFunc; // активационная функция
	Matrix* weights; // матрица весов
	double** bios; // веса смещения
	double** neurons_val; // значения нейронов
	double** neurons_err; // ошибки на нейронах
	double* neurons_bios_val; // значения нейронов смещения

public:
	void Init(data_Network data);
	void PrintConfig();
	void SetInput(double* values);
	void BackPropogation(double expect);
	void WeightsUpdater(double lr);
	void SaveWeights();
	void ReadWeights();
	double ForwardFeed();
	int SearchMaxIndex(double* value);
	void PrintValues(int L);
};

