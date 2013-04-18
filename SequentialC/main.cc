#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cmath>

#ifndef THREADS
#define THREADS 8
#endif

#define ITERATIONS 1
#define YEARS_PER_STEP 1000

#define Epsilon2 7.3890560989306495
#define GRAV_CONST 6.6738480e-11

struct double3 {
	double x, y, z;
};

struct Params {
	size_t count;
	uint32_t *hip;
	double3 *pos;
	double3 *vel;
	double *mass;
};

static void get_accelerations(Params p, size_t i, double3 * __restrict acc)
{
	const size_t count = p.count;
	double sumx = 0, sumy = 0, sumz = 0;
	double3 *a = &p.pos[i];

	#pragma omp parallel for firstprivate(count, a, p) reduction(+:sumx) \
		reduction(+:sumy) reduction(+:sumz) num_threads(THREADS)
	for (size_t j = 0; j < count; j++) {
		const double3 *b = &p.pos[j];
		double3 r;
		r.x = b->x - a->x;
		r.y = b->y - a->y;
		r.z = b->z - a->z;
		double dist = r.x * r.x + r.y * r.y + r.z * r.z + Epsilon2;
		double s = p.mass[j] / (dist * sqrt(dist));
		sumx += r.x * s;
		sumy += r.y * s;
		sumz += r.z * s;
	}
	acc->x = sumx * GRAV_CONST;
	acc->y = sumy * GRAV_CONST;
	acc->z = sumz * GRAV_CONST;
}

#ifndef NDEBUG
static void print_stars(Params p, long it)
{
	printf("Year %ld\n", it);
	for (size_t s = 0; s < p.count; s++) {
		double3 *pos = &p.pos[s];
		printf("%d %.5lf %.5lf %.5lf\n", p.hip[s], pos->x, pos->y, pos->z);
	}
}
#endif

#define dt YEARS_PER_STEP
#define hdt (dt / 2.)
#define sdt (dt * hdt)
static void simulate(Params p)
{
	const size_t count = p.count;
	double3 *acc = new double3[p.count]();
	double3 *dpos = new double3[p.count]();

	for (size_t it = 0; it < ITERATIONS; it++) {
		// Get acceleration of each star in relation with all other stars
		for (size_t i = 0; i < count; i++)
			get_accelerations(p, i, &acc[i]);
		// Move all stars
		for (size_t i = 0; i < count; i++) {
			double dx = p.vel[i].x * dt + acc[i].x * sdt;
			double dy = p.vel[i].y * dt + acc[i].y * sdt;
			double dz = p.vel[i].z * dt + acc[i].z * sdt;
			p.pos[i].x += dx;
			p.pos[i].y += dy;
			p.pos[i].z += dz;
			p.vel[i].x = dpos[i].x / dt + acc[i].x * hdt;
			p.vel[i].y = dpos[i].y / dt + acc[i].y * hdt;
			p.vel[i].z = dpos[i].z / dt + acc[i].z * hdt;
			dpos[i].x = dx;
			dpos[i].y = dy;
			dpos[i].z = dz;
		}
	}
	delete[] acc;
	delete[] dpos;
}
#undef sdt
#undef hdt
#undef dt

int main(void)
{
	struct Params p;

	int r = scanf("%zu", &p.count);

	if (r != 1) {
		printf("Error: no input.\n");
		exit(EXIT_FAILURE);
	}


	p.hip = new uint32_t[p.count];
	p.pos = new double3[p.count];
	p.vel = new double3[p.count];
	p.mass = new double[p.count];
	for (size_t i = 0; i < p.count; i++) {
		int r = scanf("%d %le %le %le %le %le %le %le", &p.hip[i], &p.pos[i].x,
				&p.pos[i].y, &p.pos[i].z, &p.vel[i].x, &p.vel[i].y,
				&p.vel[i].z, &p.mass[i]);
		if (r != 8) {
			printf("Error: invalid input.\n");
			exit(EXIT_FAILURE);
		}
	}

#ifndef NDEBUG
	printf("Starting.\n");
	simulate(p);
	print_stars(p, ITERATIONS * YEARS_PER_STEP);
#else
	simulate(p);
#endif

	delete[] p.hip;
	delete[] p.pos;
	delete[] p.vel;
	delete[] p.mass;
	return EXIT_SUCCESS;
}
