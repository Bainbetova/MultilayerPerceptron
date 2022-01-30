#pragma once
#include "ActivateFunction.h"
#include "Matrix.h"
#include <fstream>

using namespace std;

// ��� ��������� ���������� ������ ������������
struct data_Network
{
	int L; // ���������� �����
	int* size; // ���������� �������� � ������ ����
};

class Network
{
	int L; // ���������� �����
	int* size; // ���������� �������� � ������ ����
	ActivateFunction actFunc; // ������������� �������
	Matrix* weights; // ������� �����
	double** bios; // ���� ��������
	double** neurons_val; // �������� ��������
	double** neurons_err; // ������ �� ��������
	double* neurons_bios_val; // �������� �������� ��������

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

