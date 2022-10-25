
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "ForceCompute.cuh"
#include "GUI.h"
#include "Space.h"

#include <iostream>

#include <chrono>
#include <vector>


#include <omp.h>

int main()
{


	constexpr size_t n = 1000;

	std::vector<AstronomicalObject*> arrayA(n);

	for (auto& a : arrayA)
	{
		a = new Asteroid(100.f * sf::Vector2f((rand() % 2 ? -1 : 1) * rand() % 10000, (rand() % 2 ? -1 : 1) * rand() % 10000), abs(rand()) % 10000);
		a->radius = 10000;
		a->updateShape();
	}



	Saturn sat;
	sf::ContextSettings settings;
	settings.depthBits = 24;
	settings.stencilBits = 8;
	settings.antialiasingLevel = 16;
	settings.majorVersion = 3;
	settings.minorVersion = 0;
	sf::RenderWindow window(sf::VideoMode(800, 800), "Window Title", sf::Style::Default, settings);
	ImGui::SFML::Init(window);

	auto view = window.getDefaultView();
	view.setSize(sf::Vector2f(1e6, 1e6));
	window.setView(view);
	//window.setFramerateLimit(60);
	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	Space space(arrayA);


	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			GUI::Global::ProcessEvent(event);
			if (event.type == sf::Event::Closed)
				window.close();
		}


		window.clear(sf::Color::White);


		GUI::Global::Update(window);

		//space.update();

		sat.updateShape();
		sat.draw(window);


		//space.draw(window);

		GUI::Global::Render(window);


		window.display();
	}

	ImGui::SFML::Shutdown();
	return 0;
}