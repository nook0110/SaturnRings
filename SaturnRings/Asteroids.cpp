#include "Asteroids.h"

AstronomicalObject::AstronomicalObject()
{
	shape.setOrigin(sf::Vector2f(radius, radius));
	shape.setFillColor(sf::Color::Black);
	shape.setRadius(radius);
}

AstronomicalObject::AstronomicalObject(sf::Vector2f pos, float mass)
	:center_of_mass(pos), mass(mass)
{
	shape.setPosition(center_of_mass);
	shape.setOrigin(sf::Vector2f(radius, radius));
	shape.setFillColor(sf::Color::Black);
	shape.setRadius(radius);
}

void AstronomicalObject::updateShape()
{
	shape.setPosition(center_of_mass);
	shape.setOrigin(sf::Vector2f(radius, radius));
	shape.setRadius(radius);
}

void AstronomicalObject::draw(sf::RenderWindow& window) const
{
	window.draw(shape);
}

void AstronomicalObject::resetForce()
{
	forces = sf::Vector2f();
}

void AstronomicalObject::applyForce(const sf::Vector2f& force)
{
	forces += force;
}

void Asteroid::update(const sf::Time& delta)
{
	auto time = delta.asSeconds();
	auto acceleration = forces / mass;
	speed += acceleration;
	center_of_mass += speed * time;
	updateShape();
}


Saturn::Saturn()
{
	mass = SaturnMass;
	radius = SaturnRadius;
	shape.setPosition(center_of_mass);
	shape.setOrigin(sf::Vector2f(radius, radius));
	shape.setFillColor(sf::Color::Blue);
	shape.setRadius(radius);
}

void Saturn::draw(sf::RenderWindow& window) const
{
	menu.build();
	window.draw(shape);
}

