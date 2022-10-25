#pragma once
#include "Asteroids.h"
#include "ForceCompute.cuh"

class Space
{
public:
	Saturn saturn;
	std::vector<Asteroid*> asteroids;

	ForceComputer computer;

	sf::Clock globalTime;

	Space(const std::vector<AstronomicalObject*>& asteroids);

	void update();

	void draw(sf::RenderWindow& window) const;
};