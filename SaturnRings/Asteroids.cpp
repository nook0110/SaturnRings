#include "Asteroids.h"

AstronomicalObject::AstronomicalObject(sf::Vector2f pos, float mass)
	:center_of_mass(pos), mass(mass)
{
}