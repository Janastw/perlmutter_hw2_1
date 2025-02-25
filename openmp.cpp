// BECNHMARK ./serial -s 100 -n 10000

#include "common.h"
#include <cmath>
#include <vector>
#include <omp.h>


// a bin holds particle positions
typedef std::vector<std::pair<double, double>> Bin;

struct Grid {
    // 2D matrix of bins, column major order
    std::vector<Bin> bins;
    // width of square grid
    size_t grid_width;
    // width of a bin
    double bin_size;

    Grid() = default;

    // creates grid with spacing based on size of grid and cutoff of particle interaction
    Grid(const double size) {
        this->grid_width = std::floor(size / (cutoff * 2));
        this->bins = std::vector<Bin>(std::pow(this->grid_width, 2));
        this->bin_size = size / this->grid_width;
    }

    // copy constructor used for making a thread local grid with same size as global
    Grid(const Grid& other) {
        this->bins = std::vector<Bin>(other.bins.size());
        this->grid_width = other.grid_width;
        this->bin_size = other.bin_size;
    }

    ~Grid() = default;

    // get a bin from the column major order array
    Bin& at(const size_t x, const size_t y) {
        return this->bins[x * this->grid_width + y];
    }

    // get the bin a particle is in
    Bin& get_bin(const particle_t& particle) {
        return this->get_bin(particle.x, particle.y);
    }

    // get the bin for an x y coordinate
    Bin& get_bin(const double x, const double y) {
        const size_t x_bin = x / this->bin_size;
        const size_t y_bin = y / this->bin_size;

        return this->at(x_bin, y_bin);
    }
};

// create global Grid struct
Grid grid;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, const std::pair<double, double> neighbor) {
    // Calculate Distance
    const double dx = neighbor.first - particle.x;
    const double dy = neighbor.second - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    const double r = sqrt(r2);

    // Very simple short-range repulsive force
    const double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, const double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
	// initialize a grid
    grid = Grid(size);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // clear out bins
    #pragma omp for
    for (size_t i = 0; i < grid.bins.size(); i++) {
        grid.bins[i].clear();
    }

    // BEGIN PARALLEL GRID CREATION

    // make thread local grid
    Grid local_grid = grid;

    // fill thread local bin with a fraction of the particles
    #pragma omp for
    for (int i = 0; i < num_parts; i++) {
        local_grid.get_bin(parts[i]).emplace_back(parts[i].x, parts[i].y);
    }

    // calculate copying bin num stuff
    const int this_thread_num = omp_get_thread_num();
    const unsigned int batch_size = grid.bins.size() / omp_get_num_threads();
    const unsigned int num_main_batches = grid.bins.size() / batch_size;
    const unsigned int leftover_batch_size = grid.bins.size() % batch_size;

    // take turns copy batches of thread bins into different global bins
    for (unsigned int i = 0; i < num_main_batches; i++) {
        const unsigned int thread_batch_offset = (this_thread_num + i) * batch_size;
        for (unsigned int j = 0; j < batch_size; j++) {
            const unsigned int bin_num = (thread_batch_offset + j) % grid.bins.size();
            grid.bins[bin_num].insert(grid.bins[bin_num].end(), local_grid.bins[bin_num].begin(), local_grid.bins[bin_num].end());
        }
        // sycnhronize after each batch of bins
        #pragma omp barrier
    }

    // run one more time for the smaller leftover batch
    for (unsigned int i = 0; i < leftover_batch_size; i++) {
        const unsigned bin_num = ((this_thread_num + num_main_batches) * batch_size + i) % grid.bins.size();
        grid.bins[bin_num].insert(grid.bins[bin_num].end(), local_grid.bins[bin_num].begin(), local_grid.bins[bin_num].end());
    }
    #pragma omp barrier
    // END PARALLEL GRID CREATION
    
    // BEGIN SINGLE THREAD GRID CREATION
    // #pragma omp single
    // {
    //     for (int i = 0; i < num_parts; i++) {
    //         grid.get_bin(parts[i]).emplace_back(parts[i].x, parts[i].y);
    //     }
    // }
    // END SINGLE THREAD BIN CREATION

    // Compute Forces
    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;

        // get bin coordinates for this particle
        const size_t x_bin = parts[i].x / grid.bin_size;
        const size_t y_bin = parts[i].y / grid.bin_size;
        // figure out what side of the bin the particle is on
        const bool right_sided = (parts[i].x - grid.bin_size * x_bin) > (0.5 * grid.bin_size);
        const bool top_sided = (parts[i].y - grid.bin_size * y_bin) > (0.5 * grid.bin_size);
        // figure out if the particle is against any edges of the entire grid
        const bool on_right_edge = x_bin == grid.grid_width - 1;
        const bool on_left_edge = x_bin == 0;
        const bool on_top_edge = y_bin == grid.grid_width - 1;
        const bool on_bottom_edge = y_bin == 0;

        // apply forces from particles in own bin
        for (const auto neighbor : grid.at(x_bin, y_bin)) {
            apply_force(parts[i], neighbor);
        }

        // particle is in top half of its bin
        if (top_sided) {
            // particle not in a bin at top of grid
            if (!on_top_edge) {
                // apply forces with the neighbor above
                for (const auto neighbor : grid.at(x_bin, y_bin + 1)) {
                    apply_force(parts[i], neighbor);
                }
                // particle is also on the right side of its bin and not on the right edge of the grid
                if (right_sided && !on_right_edge) {
                    // apply forces from the top right bin
                    for (const auto neighbor : grid.at(x_bin + 1, y_bin + 1)) {
                        apply_force(parts[i], neighbor);
                    }
                // particle is also on the left side of its bin and not on the left edge of the grid
                } else if (!right_sided && !on_left_edge) {
                    // apply forces from the top left bin
                    for (const auto neighbor : grid.at(x_bin - 1, y_bin + 1)) {
                        apply_force(parts[i], neighbor);
                    }
                }
            }
        } else {
            if (!on_bottom_edge) {
                for (const auto neighbor : grid.at(x_bin, y_bin - 1)) {
                    apply_force(parts[i], neighbor);
                }

                if (right_sided && !on_right_edge) {
                    for (const auto neighbor : grid.at(x_bin + 1, y_bin - 1)) {
                        apply_force(parts[i], neighbor);
                    }
                } else if (!right_sided && !on_left_edge) {
                    for (const auto neighbor : grid.at(x_bin - 1, y_bin - 1)) {
                        apply_force(parts[i], neighbor);
                    }
                }
            }
        }

        // apply forces from either left or right bin
        if (right_sided && !on_right_edge) {
            for (const auto neighbor : grid.at(x_bin + 1, y_bin)) {
                apply_force(parts[i], neighbor);
            }
        } else if (!right_sided && !on_left_edge) {
            for (const auto neighbor : grid.at(x_bin - 1, y_bin)) {
                apply_force(parts[i], neighbor);
            }
        }
    }

    // Move Particles
    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}