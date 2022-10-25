#pragma once
#include <SFML/Graphics.hpp>
#include "GUI.h"


class AstronomicalObject
{
public:
	sf::Vector2f center_of_mass = sf::Vector2f();
	sf::Vector2f speed = sf::Vector2f();
	float mass = 10.f;
	float radius = 10.f;

	sf::CircleShape shape;

	sf::Vector2f forces = sf::Vector2f();
public:
	AstronomicalObject();
	AstronomicalObject(sf::Vector2f pos, float mass);

	auto getCenter() const { return center_of_mass; }
	auto getSpeed() const { return speed; }
	auto getMass() const { return mass; }

	void resetForce();
	void applyForce(const sf::Vector2f& force);
	virtual void update(const sf::Time& delta) = 0;

	void updateShape();
	virtual void draw(sf::RenderWindow& window) const;

};

class Asteroid : public AstronomicalObject
{

public:
	void update(const sf::Time& delta);

	Asteroid() = default;
	Asteroid(sf::Vector2f pos, float mass) : AstronomicalObject(pos, mass) {};

};

class Saturn : public AstronomicalObject
{
	GUI::SaturnMenu menu = GUI::SaturnMenu(&mass, &radius);
	static constexpr float SaturnMass = 5.683e26f;
	static constexpr float SaturnRadius = 58000.f;
public:
	Saturn();
	virtual void update(const sf::Time& delta) final {};
	virtual void draw(sf::RenderWindow& window) const final;
};


