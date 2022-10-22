#pragma once
#include <SFML/Graphics.hpp>

#include <list>
#include <array>
#include <vector>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Asteroids.h"

class AstronomicalObject;
class Asteroid;
class Saturn;

struct VectorGPU
{
	float x, y;

	__host__ __device__ VectorGPU(const sf::Vector2f& other = sf::Vector2f());

	__host__ __device__ VectorGPU& operator+=(const VectorGPU& other);
	__host__ __device__ VectorGPU  operator+(const VectorGPU& other) const;

	__host__ __device__ VectorGPU& operator-=(const VectorGPU& other);
	__host__ __device__ VectorGPU operator-(const VectorGPU& other) const;

	__host__ __device__ VectorGPU operator-() const;

	__host__ __device__ VectorGPU& operator*=(const float scalar);
	__host__ __device__ VectorGPU operator*(const float scalar) const;

};




class ForceComputer
{
public:
	VectorGPU* d_positions;
	std::vector<VectorGPU> h_positions;

	float* d_masses;
	std::vector<float> h_masses;

	VectorGPU* d_forces;
	std::vector<std::vector<VectorGPU>> h_forces;

	std::vector<Asteroid*> asteroids;

	ForceComputer(size_t size);
	ForceComputer(const std::vector<Asteroid*>& asteroids);

	void resize(size_t size);

	~ForceComputer();

	Asteroid* getAsteroid(size_t ind) { return asteroids[ind]; }
	const Asteroid* getAsteroid(size_t ind) const { return asteroids[ind]; }

	void findAllForces();

};


__device__ __constant__ float gravitational_constant = 6.67430e-11;

__global__
void compute(const VectorGPU* positions, const float* masses, size_t size, VectorGPU* forces)
{
	long long first = blockIdx.x * blockDim.x + threadIdx.x;
	long long second = blockIdx.y * blockDim.y + threadIdx.y;

	if (first >= size || second >= size)
	{
		return;
	}

	long long index = first * size + second;

	if (index >= size * size)
	{
		return;
	}

	auto delta = positions[second] - positions[first];
	auto distance = hypot((double)delta.x, (double)delta.y);
	forces[index] = delta;
	//forces[index] = delta * (gravitational_constant * masses[first] * masses[second] / (distance * distance * distance));
}









ForceComputer::ForceComputer(const std::vector<Asteroid*>& asteroids)
	:asteroids(asteroids)
{
	auto size = asteroids.size();
	auto pairs = size * size;

	cudaMalloc(&d_positions, size * sizeof(decltype(*d_positions)));
	cudaMalloc(&d_masses, size * sizeof(decltype(*d_masses)));
	cudaMalloc(&d_forces, pairs * sizeof(decltype(*d_forces)));

	h_positions.resize(size);
	h_masses.resize(size);
	h_forces.resize(size, std::vector<VectorGPU>(size));
}

void ForceComputer::resize(size_t size)
{
	cudaFree(d_positions);
	cudaFree(d_masses);
	cudaFree(d_forces);


	const auto pairs = size * size;
	h_positions.resize(size);
	h_masses.resize(size);
	h_forces.resize(size);
	for (auto row : h_forces)
	{
		row.resize(size);
	}
	

	cudaMalloc(&d_positions, size * sizeof(decltype(*d_positions)));
	cudaMalloc(&d_masses, size * sizeof(decltype(*d_masses)));
	cudaMalloc(&d_forces, pairs * sizeof(decltype(*d_forces)));
}

ForceComputer::~ForceComputer()
{
	cudaFree(d_positions);
	cudaFree(d_masses);
	cudaFree(d_forces);
}

void ForceComputer::findAllForces()
{
	auto size = asteroids.size();
	auto pairs = size * size;


	for (int ind = 0; ind < asteroids.size(); ++ind)
	{
		h_positions[ind] = asteroids[ind]->getCenter();
		h_masses[ind] = asteroids[ind]->getMass();
	}

	cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(decltype(*d_positions)), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_masses, h_masses.data(), h_masses.size() * sizeof(decltype(*d_masses)), cudaMemcpyKind::cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(10, 10);
	dim3 blocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

	compute << <blocks, threadsPerBlock >> > (d_positions, d_masses, size, d_forces);

	for (int row = 0; row < size; ++row)
	{
		cudaMemcpy(h_forces[row].data(), d_forces+row*size, size * sizeof(decltype(*d_forces)), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	}
}


VectorGPU::VectorGPU(const sf::Vector2f& other)
	:x(other.x), y(other.y)
{
}

__host__ __device__ VectorGPU& VectorGPU::operator+=(const VectorGPU& other)
{
	x += other.x;
	y += other.y;
	return *this;
}

__host__ __device__ VectorGPU  VectorGPU::operator+(const VectorGPU& other) const
{
	auto copy = *this;
	return (copy += other);
}

__host__ __device__ VectorGPU VectorGPU::operator-() const
{
	auto copy = *this;
	copy.x = -copy.x;
	copy.y = -copy.y;
	return copy;
}

__host__ __device__ VectorGPU& VectorGPU::operator*=(const float scalar)
{
	x *= scalar;
	y *= scalar;
	return *this;
}

__host__ __device__ VectorGPU VectorGPU::operator*(const float scalar) const
{
	auto copy = *this;
	copy *= scalar;
	return copy;
}

__host__ __device__ VectorGPU& VectorGPU::operator-=(const VectorGPU& other)
{
	(*this) += (-other);
	return *this;
}

__host__ __device__ VectorGPU  VectorGPU::operator-(const VectorGPU& other) const
{
	auto copy = *this;
	copy -= other;
	return copy;
}

