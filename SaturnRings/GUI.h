#pragma once

#include "SFML/Graphics.hpp"

#include "imgui.h"
#include "imgui-SFML.h"

namespace GUI
{

	class Global
	{
		static sf::Clock deltaClock;
	public:
		static void Update(sf::RenderWindow& window);
		static void Render(sf::RenderWindow& window);
		static void ProcessEvent(sf::Event event);
	};

	class SaturnMenu
	{
		float* mass;
		float* radius;
		const float min_radius = 0.f;
		const float max_radius = 100000.f;
	public:
		SaturnMenu(float* mass, float* radius)
			:mass(mass), radius(radius) {}
		void build() const;
	};
}