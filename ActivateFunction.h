#pragma once
#include <iostream>
/* Функции активации */

// будем использовать 3 функции активации: сигмоида, линейный выпрямитель с утечкой, гиперболический тангенс
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