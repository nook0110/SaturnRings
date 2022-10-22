
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "ForceCompute.cuh"

#include <iostream>

#include <vector>

int main()
{
	constexpr size_t n = 1000;

	std::vector<Asteroid*> arrayA(n);

	for (auto& a : arrayA)
	{
		a = new Asteroid();
		a->center_of_mass =sf::Vector2f(rand() % 100, rand() % 100);
		a->mass = rand() % 10000;
	}

	auto x = new ForceComputer(arrayA);


	x->findAllForces();

	x->resize(10);

	for (auto a : arrayA)
	{
		delete a;;
	}

	return 0;
}