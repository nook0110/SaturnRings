#pragma once
#include <SFML/Graphics.hpp>

class AstronomicalObject
{
public:
	sf::Vector2f center_of_mass = sf::Vector2f();
	sf::Vector2f speed = -center_of_mass;
	float mass = 10.f;


public:
	AstronomicalObject() = default;
	AstronomicalObject(sf::Vector2f pos, float mass);

	auto [[nodiscard]] getCenter() const { return center_of_mass; }
	auto [[nodiscard]] getSpeed() const { return speed; }
	auto [[nodiscard]] getMass() const { return mass; }

};

class Asteroid : public AstronomicalObject
{
public:
	Asteroid() = default;
	Asteroid(sf::Vector2f pos, float mass) : AstronomicalObject(pos, mass) {};

};

class Saturn : public AstronomicalObject
{
public:
	//Saturn() = default;
};



