#include "GUI.h"

#include <iostream>

using namespace GUI;

sf::Clock Global::deltaClock;

void Global::Update(sf::RenderWindow& window)
{
	ImGui::SFML::Update(window, deltaClock.restart());
}

void  Global::Render(sf::RenderWindow& window)
{
	ImGui::SFML::Render(window);
}

void Global::ProcessEvent(sf::Event event)
{
	ImGui::SFML::ProcessEvent(event);
}

void GUI::SaturnMenu::build() const
{
	ImGui::Begin("Satrurn settings");
	ImGui::InputFloat("Mass (kg)", mass, 0,0, "%e");
	ImGui::SliderFloat("Radius (km) ", radius, min_radius, max_radius);
	ImGui::End();
}
