#pragma once
#include <iostream>
/* ������� ��������� */

// ����� ������������ 3 ������� ���������: ��������, �������� ����������� � �������, ��������������� �������
enum activateFunc { singmoid = 1, ReLU, thx };

class ActivateFunction
{
	activateFunc actFunc;
public:
	void set();
	void use(double* value, int n);
	void useDer(double* value, int n);
	double useDer(double value);
};