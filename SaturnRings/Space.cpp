#include "Space.h"

Space::Space(const std::vector<AstronomicalObject*>& asteroids)
	:computer(asteroids)
{
	globalTime.restart();
}

void Space::update()
{
	auto& forces = computer();

	auto delta = globalTime.getElapsedTime();
	globalTime.restart();

	for (int asteroid = 0; asteroid < asteroids.size(); ++asteroid)
	{
		asteroids[asteroid]->resetForce();
		asteroids[asteroid]->applyForce(sf::Vector2f(forces[asteroid].x, forces[asteroid].y));
		asteroids[asteroid]->update(delta);
	}

}

void Space::draw(sf::RenderWindow& window) const
{
	for (auto asteroid : asteroids)
	{
		asteroid->draw(window);
	}
}
