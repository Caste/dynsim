__author__ = 'chaubold'

import numpy as np
import numpy.linalg
import OpenGL.GL as gl
import OpenGL.GLUT as glut


def crossProdMatrix(vec):
    return np.array([[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0]])

# Bringt kleine Abweichungen von der Orthonormalitaet
# von M in Ordnung, ein Durchlauf genuegt, super Algorithmus!
# Wenn bei Nutation "orthonormalize" nicht angewandt wird, so Desaster.
def orthonormalize(m):
    eps2 = 1.0e-18
    v1 = m[:][0]
    v2 = m[:][1]
    v3 = m[:][2]

    v1 /= numpy.linalg.norm(v1)
    v2 /= numpy.linalg.norm(v2)
    v3 /= numpy.linalg.norm(v3)

    for i in xrange(10):
        stop = True
        v12 = np.cross(v1, v2)
        if numpy.linalg.norm(v12 - v3)**2 > eps2:
            stop = False

        v31 = np.cross(v3, v1)
        if numpy.linalg.norm(v31 - v2)**2 > eps2:
            stop = False

        v23 = np.cross(v2, v3)
        if numpy.linalg.norm(v23 - v1)**2 > eps2:
            stop = False

        v1 = (v1 + v23) * 0.5
        v2 = (v2 + v31) * 0.5
        v3 = (v3 + v12) * 0.5

        if stop:
            break

    result = np.zeros((3, 3))
    result[:][0] = v1
    result[:][1] = v2
    result[:][2] = v3

    return result

#def orthonormalize(matrix):
#    q, r = numpy.linalg.qr(matrix)
#    return q

class rigid_body:
    damping = 0.05

    def __init__(self):
        # for display
        self.color = np.array([0.0, 0.0, 0.0])

        # translational
        self.mass = 0.0
        self.position = np.array([0.0, 0.0, 0.0])
        self._velocity = np.array([0.0, 0.0, 0.0])
        self._force = np.array([0.0, 0.0, 0.0])

        # rotational
        self._orientation = np.identity(3)
        self._angular_velocity = np.array([0.0, 0.0, 0.0])
        self._torque = np.array([0.0, 0.0, 0.0])
        self._inertia_tensor = np.identity(3)

    def time_step(self, delta_time, use_runge_kutta):
        # do not simulate bodies without mass
        if self.mass == 0.0:
            return

        # choose appropriate method
        if use_runge_kutta:
            self.__time_step_rk(delta_time)
        else:
            self.__time_step_euler(delta_time)

    def __time_step_euler(self, delta_time):
        # update translations
        self.position += self._velocity * delta_time
        self._velocity += (self._force / self.mass) * delta_time
        self._force = np.array([0.0, 0.0, 0.0])

        # update rotations (np.dot is matrix multiplication, matrix-vector multiplication or dot product for 2 vectors)
        self._orientation += np.dot(crossProdMatrix(self._angular_velocity), self._orientation) * delta_time
        self._angular_velocity += np.dot(numpy.linalg.inv(self._inertia_tensor),
                                        (self._torque - np.dot(self._angular_velocity,
                                                              np.dot(self._inertia_tensor,
                                                                     self._angular_velocity)))) * delta_time
        self._torque = np.array([0.0, 0.0, 0.0])

    def __acceleration(self, pos, vel, t):
        # this could include constraints later!
        return self._force / self.mass - rigid_body.damping * vel

    def __angular_acceleration(self, orientation, angular_velocity, t):
        # split up w' = J-1 * (t - w x (Jw))
        # into     w' = J-1 * (t - w x (a))
        #               J-1 * (t - b)
        #               J-1 * c

        a = np.dot(self._inertia_tensor, angular_velocity)
        b = np.dot(crossProdMatrix(angular_velocity), a)
        c = self._torque - b

        return np.dot(numpy.linalg.inv(self._inertia_tensor), c) - rigid_body.damping * angular_velocity

    def __time_step_rk(self, delta_time):
        # translational
        x1 = self.position
        v1 = self._velocity
        a1 = self.__acceleration(x1, v1, 0)

        x2 = self.position + 0.5 * v1 * delta_time
        v2 = self._velocity + 0.5 * a1 * delta_time
        a2 = self.__acceleration(x2, v2, delta_time / 2.0)

        x3 = self.position + 0.5 * v2 * delta_time
        v3 = self._velocity + 0.5 * a2 * delta_time
        a3 = self.__acceleration(x3, v3, delta_time / 2.0)

        x4 = self.position + v3 * delta_time
        v4 = self._velocity + a3 * delta_time
        a4 = self.__acceleration(x4, v4, delta_time)

        self.position += (delta_time / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        self._velocity += (delta_time / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        self._force = np.array([0.0, 0.0, 0.0])

        # rotational
        o1 = self._orientation
        w1 = self._angular_velocity
        t1 = self.__angular_acceleration(o1, w1, 0)

        # orientation update is R' = w x R, where we need the cross product matrix of w!
        o2 = self._orientation + np.dot(crossProdMatrix(w1), o1) * 0.5 * delta_time
        w2 = self._angular_velocity + 0.5 * t1 * delta_time
        t2 = self.__angular_acceleration(o2, w2, delta_time / 2.0)

        o3 = self._orientation + np.dot(crossProdMatrix(w2), o2) * 0.5 * delta_time
        w3 = self._angular_velocity + 0.5 * t2 * delta_time
        t3 = self.__angular_acceleration(o3, w3, delta_time / 2.0)

        o4 = self._orientation + np.dot(crossProdMatrix(w3), o3) * 0.5 * delta_time
        w4 = self._angular_velocity + t3 * delta_time
        t4 = self.__angular_acceleration(o4, w4, delta_time)

        self._orientation += (delta_time / 6.0) * np.dot(crossProdMatrix(w1 + 2 * w2 + 2 * w3 + w4), self._orientation)
        self._angular_velocity += (delta_time / 6.0) * (t1 + 2 * t2 + 2 * t3 + t4)
        self._torque = np.array([0.0, 0.0, 0.0])

        # check orthonormality of orientation: should be Identity matrix!!
        print(np.dot(np.transpose(self._orientation), self._orientation))

        # orthonormalize orientation
        self._orientation = orthonormalize(self._orientation)

    def draw(self):
        print("Not implemented!")

    def apply_force(self, force):
        self._force += force

    def convert_to_local(self, point):
        return np.dot(numpy.linalg.inv(self._orientation), (point - self.position))

    def convert_to_global(self, point):
        return np.dot(self._orientation, point) + self.position

    def velocity_of_point(self, point):
        local_point = point - self.position
        velocity = self._velocity + np.dot(crossProdMatrix(self._angular_velocity), local_point)
        return velocity

    def apply_force_at(self, force, point):
        local_point = self.convert_to_local(point)

        # if force is not applied at center: compute torque
        self._torque += np.dot(crossProdMatrix(local_point), force)

        # apply force to linear motion of body
        self.apply_force(force)

class cube(rigid_body):
    def __init__(self):
        rigid_body.__init__(self)
        self.width = 1.0
        self.height = 1.0
        self.depth = 1.0

        self._inertia_tensor = np.array([[(self.height**2 + self.depth**2)/12.0, 0.0, 0.0],
            [0.0, (self.width**2+self.depth**2)/12.0, 0.0], [0.0, 0.0, (self.width**2 + self.height**2)/12.0]])

    def convert_to_local(self, point):
        # get the rotated and translated point in local coordinates and apply inverse scaling
        local_point = rigid_body.convert_to_local(self, point)
        local_point[0] /= self.width
        local_point[1] /= self.height
        local_point[2] /= self.depth
        return local_point

    def convert_to_global(self, point):
        # apply scaling, then rotation and translation
        local_point = point
        local_point[0] *= self.width
        local_point[1] *= self.height
        local_point[2] *= self.depth
        return rigid_body.convert_to_global(self, local_point)

    def draw(self):
        # store the matrix that has been used for transformations before,
        # because all translations, rotations, scalings alter it
        gl.glPushMatrix()

        # set the appropriate color
        gl.glColor3f(self.color[0], self.color[1], self.color[2])

        # translate, rotate and scale operations need to be read from last to first,
        # in the same way as a multiplication of the vector through matrices would be:
        # pointToRender = Translation * Rotation * Scaling * inputPoint

        # apply the translation
        gl.glTranslatef(self.position[0], self.position[1], self.position[2])

        # apply the rotation matrix (needs to be 4x4)
        rotation_matrix = np.zeros((4, 4))
        rotation_matrix[:3, :3] = np.transpose(self._orientation)
        rotation_matrix[3, 3] = 1.0

        gl.glMultMatrixf(rotation_matrix)

        # scale to appropriate size
        gl.glScalef(self.width, self.height, self.depth)

        # draw cube
        glut.glutSolidCube(1.0)

        # restore previous transformation matrix
        gl.glPopMatrix()
