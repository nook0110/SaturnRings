#pragma once
#include <SFML/Graphics.hpp>

#include <list>
#include <array>

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



template<size_t size>
class ForceComputer
{
public:
	VectorGPU* d_positions;
	std::array<VectorGPU, size> h_positions;

	float* d_masses;
	std::array<float, size> h_masses;

	const size_t pairs = size * size;
	VectorGPU* d_forces;
	std::array<std::array< VectorGPU, size>, size> h_forces;

	std::array<Asteroid, size> asteroids;

	ForceComputer();
	ForceComputer(const std::array<Asteroid, size>& asteroids);

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

	auto delta = positions[second] - positions[first];
	auto distance = hypot((double)delta.x, (double)delta.y);
	forces[index] = delta * (gravitational_constant * masses[first] * masses[second] / (distance * distance * distance));
}








template<size_t size>
ForceComputer<size>::ForceComputer()
{
	
}

template<size_t size>
ForceComputer<size>::ForceComputer(const std::array<Asteroid, size>& asteroids)
	:asteroids(asteroids)
{
	
}

template<size_t size>
void ForceComputer<size>::findAllForces()
{

	cudaMalloc(&d_positions, size * sizeof(decltype(*d_positions)));
	cudaMalloc(&d_masses, size * sizeof(decltype(*d_masses)));
	cudaMalloc(&d_forces, pairs * sizeof(decltype(*d_forces)));

	float* x = (float*)malloc(size * sizeof(float));
	float* y = (float*)malloc(size * sizeof(float));


	for (int ind = 0; ind < asteroids.size(); ++ind)
	{
		h_positions[ind] = asteroids[ind].getCenter();
		h_masses[ind] = asteroids[ind].getMass();
		x[ind] = 100.f;
	}

	cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(decltype(*d_positions)), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_masses, h_masses.data(), h_masses.size() * sizeof(decltype(*d_masses)), cudaMemcpyKind::cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(10, 10);
	dim3 blocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

	compute<<<blocks, threadsPerBlock>>>(d_positions, d_masses, size, d_forces);

	cudaMemcpy(h_forces.data(), d_forces, pairs * sizeof(decltype(*d_forces)), cudaMemcpyKind::cudaMemcpyDeviceToHost);
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

