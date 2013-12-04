__author__ = 'chaubold'

import math

import OpenGL.GLUT as glut
import OpenGL.GL as gl
import OpenGL.GLU as glu
import numpy as np

import rigidbody
import spring
import cloth


bodies = []
springs = []
blanket = cloth.cloth(3, 3, 1.0, 5.0)

# -------------------------------------------------------
# camera parameters

# angle of rotation for the camera direction
angle_x = 0.0
angle_y = 0.0

# actual vector representing the camera's direction
look_at_x = 0.0
look_at_y = 0.0
look_at_z = -1.0

# XZ position of the camera
cam_pos_x = 0.0
cam_pos_y = -3.0
cam_pos_z = 25.0

# the key states. These variables will be zero
# when no key is being presses
deltaAngle_x = 0.0
deltaAngle_y = 0.0
deltaMove = 0
xOrigin = -1
yOrigin = -1

# timer
lastTime = 0
maxTimeStep = 0.01
use_runge_kutta = True

# -------------------------------------------------------
def computePos(deltaMove):
    global cam_pos_x
    global cam_pos_z
    global cam_pos_y

    cam_pos_x += deltaMove * look_at_x * 0.1
    cam_pos_y += deltaMove * look_at_y * 0.1
    cam_pos_z += deltaMove * look_at_z * 0.1

# -------------------------------------------------------
def render():
    # clear screen
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # reset camera
    gl.glLoadIdentity()

    # perform camera movement if necessary
    if deltaMove:
        computePos(deltaMove)

    # set up camera
    glu.gluLookAt(cam_pos_x, cam_pos_y, cam_pos_z, cam_pos_x+look_at_x, cam_pos_y+look_at_y,  cam_pos_z+look_at_z, 0.0, 1.0,  0.0)

    # draw all bodies
    for body in bodies:
        body.draw()

    # draw all springs
    for s in springs:
        s.draw()

    blanket.draw()

    # redraw
    glut.glutSwapBuffers()
    glut.glutPostRedisplay()

# -------------------------------------------------------
def init():

    # set up viewing parameters
    gl.glMatrixMode(gl.GL_PROJECTION)
    glu.gluPerspective(60.0, 1.0, 1.0, 100.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # set wireframe mode
    #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

    # set up lighting
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_COLOR_MATERIAL)
    gl.glEnable(gl.GL_LIGHT0)
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [-5.0, 4.0, 2.0])

    # time counter
    global lastTime
    lastTime = glut.glutGet(glut.GLUT_ELAPSED_TIME)

    # set up a few bodies to simulate
    c = rigidbody.cube()
    c.position = np.array([2.0, 0.0, 0.0])
    c.mass = 0.0
    c.color = np.array([1.0, 0.0, 0.0])
    c.apply_force(np.array([0.00, 30.0, 0.0]))
    c.apply_force_at(np.array([0.0, 10.0, 0.0]), np.array([2.1, -0.1, 0.1]))
    bodies.append(c)

    c2 = rigidbody.sphere(0.5)
    c2.position = np.array([-2.0, -3.0, 0.0])
    c2.mass = 1.0
    c2.color = np.array([0.0, 0.0, 1.0])
    #c2.apply_force_at(np.array([0.0, 50.0, 0.0]), np.array([-2.1, -0.1, 0.0]))
    bodies.append(c2)

    c3 = rigidbody.cube()
    c3.position = np.array([-4.0, -1.0, 0.0])
    c3.mass = 3.0
    c3.color = np.array([0.0, 0.0, 1.0])
    bodies.append(c3)

    #s = spring.spring(2.0, c, np.array([1.5, 0.5, 0.5]), c2, np.array([-1.5, 0.5, 0.5]))
    s = spring.spring(5.0, c, np.array([2.0, -0.5, 0.0]), c2, np.array([-2.1, -2.5, 0.2]))
    s.length = 2.0
    springs.append(s)

    s2 = spring.spring(5.0, c2, c2.position, c3, c3.position + np.array([0.3, 0.5, 0.4]))
    springs.append(s2)

# -------------------------------------------------------
def simulate():
    # simulate time between current and last frame, if necessary by splitting up into smaller time steps
    global lastTime
    timeStep = glut.glutGet(glut.GLUT_ELAPSED_TIME) - lastTime
    simulationTime = timeStep / 1000.0

    gravity = np.array([0.0, -9.81, 0.0])

    while simulationTime > 0:
        # apply spring forces
        for s in springs:
            s.apply_forces_to_bodies()

        # simulate bodies and add gravity
        for body in bodies:
            body.apply_force(gravity)
            body.time_step(min(simulationTime, maxTimeStep), use_runge_kutta)

        blanket.simulate_timestep(min(simulationTime, maxTimeStep), gravity, use_runge_kutta)

        simulationTime -= maxTimeStep

    lastTime += timeStep

# -------------------------------------------------------
def mouseMove(mx, my):
    global deltaAngle_x
    global deltaAngle_y
    global look_at_x
    global look_at_y
    global look_at_z

    if xOrigin >= 0:
        deltaAngle_x = (mx - xOrigin) * -0.005

    if yOrigin >= 0:
        deltaAngle_y = (my - yOrigin) * 0.005

    look_at_x = math.sin(angle_x + deltaAngle_x) * math.cos(deltaAngle_y + angle_y)
    look_at_y = math.sin(deltaAngle_y + angle_y)
    look_at_z = -math.cos(angle_x + deltaAngle_x) * math.cos(deltaAngle_y + angle_y)

# -------------------------------------------------------
def mouseButton(button, state, bx, by):
    global angle_x
    global angle_y
    global xOrigin
    global yOrigin

    # only start motion if the left button is pressed
    if button == glut.GLUT_LEFT_BUTTON:
        # when the button is released
        if state == glut.GLUT_UP:
            angle_x += deltaAngle_x
            xOrigin = -1

            angle_y += deltaAngle_y
            yOrigin = -1
        else:
            # state == GLUT_DOWN
            xOrigin = bx
            yOrigin = by

# -------------------------------------------------------
def pressKey(key, xx, yy):
    global deltaMove

    if key == glut.GLUT_KEY_UP:
        deltaMove = 0.5
    elif key == glut.GLUT_KEY_DOWN:
        deltaMove = -0.5

# -------------------------------------------------------
def releaseKey(key, xx, yy):
    global deltaMove

    deltaMove = 0

# -------------------------------------------------------
# main program init
glut.glutInit()
glut.glutInitWindowSize(800, 600)
glut.glutInitDisplayMode(glut.GLUT_RGB | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
glut.glutCreateWindow("DynSimTest")
glut.glutDisplayFunc(render)
glut.glutIdleFunc(simulate)
glut.glutMouseFunc(mouseButton)
glut.glutMotionFunc(mouseMove)
glut.glutSpecialFunc(pressKey)
glut.glutSpecialUpFunc(releaseKey)

init()

glut.glutMainLoop()