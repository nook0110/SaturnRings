
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "ForceCompute.cuh"

#include <iostream>



int main()
{
	std::array<Asteroid, 3> arrayA = { Asteroid(sf::Vector2f(10,10), (float)30000),
		Asteroid(sf::Vector2f(30,10), (float)30000),
		Asteroid(sf::Vector2f(10,30), (float)30000) };
	auto x = ForceComputer<3>(arrayA);

	x.findAllForces();


	for (auto f : x.h_forces)
	{
		for (auto f2 : f)
		{
			std::cout << f2.x << " " << f2.y << "   ";
		}
		std::cout << "\n";
	}


	return 0;
}