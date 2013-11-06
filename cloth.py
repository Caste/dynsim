__author__ = 'chaubold'

import rigidbody
import spring
import numpy as np

class cloth:
    def __init__(self, grid_width_x, grid_width_y, particle_mass, spring_stiffness):
        self.__particles = []
        self.__springs = []
        self.__width_x = grid_width_x
        self.__width_y = grid_width_y
        self.__grid_spacing = 1.0

        self.__particle_mass = particle_mass
        self.__spring_stiffness = spring_stiffness

        self.__generate_grid()
        self.__generate_springs()
        self.__generate_springs_diagonal()
        self.__generate_springs_bending()

        # set two particle to have zero mass
        self.__particles[0][0].mass = 0.0
        self.__particles[-1][0].mass = 0.0


    def __generate_grid(self):
        for x in xrange(self.__width_x):
            particle_row = []
            for y in xrange(self.__width_y):
                p = rigidbody.sphere(0.05)
                p.mass = self.__particle_mass
                p.position = np.array([self.__grid_spacing * x, 0.0, self.__grid_spacing * y])
                particle_row.append(p)

            self.__particles.append(particle_row)

    def __generate_springs(self):
        # generate grid without the last row/column
        for x in xrange(self.__width_x - 1):
            for y in xrange(self.__width_y - 1):
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x][y],
                                                    self.__particles[x][y].position,
                                                    self.__particles[x+1][y],
                                                    self.__particles[x+1][y].position))
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x][y],
                                                    self.__particles[x][y].position,
                                                    self.__particles[x][y+1],
                                                    self.__particles[x][y+1].position))

        # last row
        for x in xrange(self.__width_x - 1):
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[x  ][self.__width_y - 1],
                                                self.__particles[x  ][self.__width_y - 1].position,
                                                self.__particles[x+1][self.__width_y - 1],
                                                self.__particles[x+1][self.__width_y - 1].position))

        # last column
        for y in xrange(self.__width_y - 1):
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[self.__width_x - 1][y],
                                                self.__particles[self.__width_x - 1][y].position,
                                                self.__particles[self.__width_x - 1][y+1],
                                                self.__particles[self.__width_x - 1][y+1].position))

    def __generate_springs_diagonal(self):
        for x in xrange(self.__width_x - 1):
            for y in xrange(self.__width_y - 1):
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x][y],
                                                    self.__particles[x][y].position,
                                                    self.__particles[x+1][y+1],
                                                    self.__particles[x+1][y+1].position))
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x+1][y],
                                                    self.__particles[x+1][y].position,
                                                    self.__particles[x][y+1],
                                                    self.__particles[x][y+1].position))

    def __generate_springs_bending(self):
        for x in xrange(self.__width_x - 2):
            for y in xrange(self.__width_y - 2):
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x][y],
                                                    self.__particles[x][y].position,
                                                    self.__particles[x][y+2],
                                                    self.__particles[x][y+2].position))
                self.__springs.append(spring.spring(self.__spring_stiffness,
                                                    self.__particles[x][y],
                                                    self.__particles[x][y].position,
                                                    self.__particles[x+2][y],
                                                    self.__particles[x+2][y].position))

        # last rows
        for x in xrange(self.__width_x - 2):
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[x  ][self.__width_y - 2],
                                                self.__particles[x  ][self.__width_y - 2].position,
                                                self.__particles[x+2][self.__width_y - 2],
                                                self.__particles[x+2][self.__width_y - 2].position))
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[x  ][self.__width_y - 1],
                                                self.__particles[x  ][self.__width_y - 1].position,
                                                self.__particles[x+2][self.__width_y - 1],
                                                self.__particles[x+2][self.__width_y - 1].position))

        # last columns
        for y in xrange(self.__width_y - 2):
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[self.__width_x - 2][y],
                                                self.__particles[self.__width_x - 2][y].position,
                                                self.__particles[self.__width_x - 2][y+2],
                                                self.__particles[self.__width_x - 2][y+2].position))
            self.__springs.append(spring.spring(self.__spring_stiffness,
                                                self.__particles[self.__width_x - 2][y],
                                                self.__particles[self.__width_x - 2][y].position,
                                                self.__particles[self.__width_x - 2][y+2],
                                                self.__particles[self.__width_x - 2][y+2].position))

    def simulate_timestep(self, delta_time, gravity, use_runge_kutta):
        # simulate springs
        for s in self.__springs:
            s.apply_forces_to_bodies()

        # simulate particles
        for row in self.__particles:
            for p in row:
                p.apply_force(gravity)
                p.time_step(delta_time, use_runge_kutta)

    def draw(self):
        # simulate springs
        for s in self.__springs:
            s.draw()

        # simulate particles
        for row in self.__particles:
            for p in row:
                p.draw()