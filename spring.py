__author__ = 'chaubold'

import rigidbody
import numpy as np
import numpy.linalg as la
import OpenGL.GL as gl

class spring:
    def __init__(self, stiffness, body1, global_contact_point1, body2, global_contact_point2, color=None):
        self.stiffness = stiffness
        self.__length = la.norm(global_contact_point1 - global_contact_point2)

        # bodies that are connected by this spring
        # together with the local point where the connection occurs.
        # this point can be transformed to the global system to compute the spring length and apply forces
        self.body1 = body1
        self.__contact_point1 = body1.convert_to_local(global_contact_point1)

        self.body2 = body2
        self.__contact_point2 = body2.convert_to_local(global_contact_point2)

        self.color = color
        if not self.color:
            self.color = np.array([1.0, 0.0, 1.0])

    def apply_forces_to_bodies(self):
        # compute current positions of contact points
        point1 = self.body1.convert_to_global(self.__contact_point1)
        point2 = self.body2.convert_to_global(self.__contact_point2)

        # compute current length
        current_length = la.norm(point1 - point2)

        # compute force properties
        force_direction = (point1 - point2) / current_length
        force_strength = self.stiffness * (current_length - self.__length)

        # get point velocities:
        point1_velocity = self.body1.velocity_of_point(point1)
        point2_velocity = self.body2.velocity_of_point(point2)

        # add damping to the force
        damping = 0.0
        force = force_direction * force_strength - damping * (point2_velocity - point1_velocity)

        self.body2.apply_force_at(force, point2)
        self.body1.apply_force_at(force * -1.0, point1)
        #self.body2.apply_force(force)
        #self.body1.apply_force(force * -1.0)

    def draw(self):
        gl.glPushMatrix()

        # get global points
        point1 = self.body1.convert_to_global(self.__contact_point1)
        point2 = self.body2.convert_to_global(self.__contact_point2)

        # draw a line between them
        gl.glColor3f(self.color[0], self.color[1], self.color[2])
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(point1[0], point1[1], point1[2])
        gl.glVertex3f(point2[0], point2[1], point2[2])
        gl.glEnd()

        gl.glPopMatrix()


