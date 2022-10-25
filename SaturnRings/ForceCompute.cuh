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

	__host__ __device__ VectorGPU& operator/=(const float scalar);
	__host__ __device__ VectorGPU operator/(const float scalar) const;
};




class ForceComputer
{
public:
	VectorGPU* d_positions;
	std::vector<VectorGPU> h_positions;

	float* d_masses;
	std::vector<float> h_masses;

	VectorGPU* d_forces;
	std::vector<VectorGPU> h_forces;

	std::vector<AstronomicalObject*> asteroids;

	ForceComputer(size_t size);
	ForceComputer(const std::vector<AstronomicalObject*>& asteroids);

	void resize(size_t size);

	~ForceComputer();

	AstronomicalObject* getAsteroid(size_t ind) { return asteroids[ind]; }
	const AstronomicalObject* getAsteroid(size_t ind) const { return asteroids[ind]; }

	const std::vector<VectorGPU>& operator()();
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

	auto delta = positions[second] - positions[first];
	auto distance = hypotf(delta.x, delta.y);
	forces[first] += (delta / distance) * (gravitational_constant * powf((masses[first] / distance), 2));
}









ForceComputer::ForceComputer(const std::vector<AstronomicalObject*>& asteroids)
	:asteroids(asteroids)
{
	auto size = asteroids.size();

	cudaMalloc(&d_positions, size * sizeof(decltype(*d_positions)));
	cudaMalloc(&d_masses, size * sizeof(decltype(*d_masses)));
	cudaMalloc(&d_forces, size * sizeof(decltype(*d_forces)));

	h_positions.resize(size);
	h_masses.resize(size);
	h_forces.resize(size);
}

void ForceComputer::resize(size_t size)
{
	cudaFree(d_positions);
	cudaFree(d_masses);
	cudaFree(d_forces);

	h_positions.resize(size);
	h_masses.resize(size);
	h_forces.resize(size);


	cudaMalloc(&d_positions, size * sizeof(decltype(*d_positions)));
	cudaMalloc(&d_masses, size * sizeof(decltype(*d_masses)));
	cudaMalloc(&d_forces, size * sizeof(decltype(*d_forces)));
}

ForceComputer::~ForceComputer()
{
	cudaFree(d_positions);
	cudaFree(d_masses);
	cudaFree(d_forces);
}

const std::vector<VectorGPU>& ForceComputer::operator()()
{
	auto size = asteroids.size();


	for (int ind = 0; ind < asteroids.size(); ++ind)
	{
		h_positions[ind] = asteroids[ind]->getCenter();
		h_masses[ind] = asteroids[ind]->getMass();
	}


	cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(decltype(*d_positions)), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_masses, h_masses.data(), h_masses.size() * sizeof(decltype(*d_masses)), cudaMemcpyKind::cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(32, 32);
	dim3 blocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x, (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

	compute << <blocks, threadsPerBlock >> > (d_positions, d_masses, size, d_forces);

	cudaMemcpy(h_forces.data(), d_forces, size * sizeof(decltype(*d_forces)), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	return h_forces;
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

__host__ __device__ VectorGPU& VectorGPU::operator/=(const float scalar)
{
	x /= scalar;
	y /= scalar;
	return *this;
}

__host__ __device__ VectorGPU VectorGPU::operator/(const float scalar) const
{
	auto copy = *this;
	copy /= scalar;
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

