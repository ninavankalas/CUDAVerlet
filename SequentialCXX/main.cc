#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <cmath>

namespace constants {

const double e = 2.71828182845904523536;
const double G = 6.6738480e-11;

}

namespace {

struct double3 {
	double x, y, z;

	double3() : x(0), y(0), z(0) {}

	double3(double x, double y, double z) : x(x), y(y), z(z) {}

	double3(const double3& b) : x(b.x), y(b.y), z(b.z) {}

	~double3() {}

	friend std::istream& operator >>(std::istream& stream, double3& s) {
		return stream >> s.x >> s.y >> s.z;
	}

	inline
	double3& operator =(const double3& b) {
		x = b.x;
		y = b.y;
		z = b.z;
		return *this;
	}

	inline
	double3 operator +=(const double3& b) {
		*this  = *this + b;
		return *this;
	}

	inline
	const double3 operator -(const double3& b) const {
		return double3(x - b.x, y - b.y, z - b.z);
	}

	inline
	const double3 operator +(const double3& b) const {
		return double3(x + b.x, y + b.y, z + b.z);
	}

	inline
	const double3 operator /(double scalar) const {
		return double3(x / scalar, y / scalar, z / scalar);
	}

	inline
	const double3 operator *(double scalar) const {
		return double3(x * scalar, y * scalar, z * scalar);
	}

	inline
	double distance(void) const {
		return x * x + y * y + z * z + (constants::e * constants::e);
	}

	std::string str(void) const {
		std::stringstream ss;
		ss << std::setprecision(8) << x << " " << y << " " << z;
		return ss.str();
	}
};

class Star {
public:
	uint32_t hip = 0;
	double3 pos = double3(0, 0, 0);
	double3 vel = double3(0, 0, 0);
	double3 acceleration = double3(0, 0, 0);
	double3 prev_delta = double3(0, 0, 0);
	double mass = 0;

	friend std::istream& operator >>(std::istream& stream, Star& s) {
		return stream >> s.hip >> s.pos >> s.vel >> s.mass;
	}

	inline
	void update_acceleration(const std::vector<Star>& __restrict stars) {
		double3 a(0, 0, 0);
		for (auto& star: stars) {
			double3 r(star.pos - pos);
			double dist = r.distance();

			a += r * (star.mass / (dist * sqrt(dist)));
		}
		acceleration = a * constants::G;
	}

	void move_star(double dt) {
		const double3 delta = vel * dt + acceleration * ((dt * dt) / 2.);
		pos += delta;
		vel = prev_delta / dt + acceleration * (dt / 2.);
		prev_delta = delta;
	}

	std::string str(void) const {
		std::stringstream ss;
		ss << pos.str() << mass;
		return ss.str();
	}
};

template <size_t iterations, int elapsed_time_per_iteration>
static void simulate(std::vector<Star>& __restrict stars)
{
	for (size_t it = 0; it < iterations; it++) {
		for (auto& s: stars)
			s.update_acceleration(stars);

		for (auto& s: stars)
			s.move_star(static_cast<double>(elapsed_time_per_iteration));
	}
}

}

int main(void)
{
	size_t count;
	if (scanf("%zu", &count) != 1) {
		printf("Error: no input.\n");
		exit(EXIT_FAILURE);
	}
	std::vector<Star> stars;
	stars.resize(count);
	for (size_t i = 0; i < count; i++) {
		Star& s = stars[i];
		int r = scanf("%u %le %le %le %le %le %le %le", &s.hip, &s.pos.x,
				&s.pos.y, &s.pos.z, &s.vel.x, &s.vel.y,	&s.vel.z, &s.mass);
		if (r != 8) {
			printf("Error: invalid input.\n");
			exit(EXIT_FAILURE);
		}

	}
	simulate<10, 1000>(stars);
#ifndef NDEBUG
	for (auto& s: stars)
		std::cout << s.str() << std::endl;
#endif
	return EXIT_SUCCESS;
}
