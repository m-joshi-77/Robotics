# This code was written in Python in the google colab notebook instance.
# The mr.IKinBody() was referred to write this code.

#!pip install modern_robotics # Installling the modern robotics in the colab notebook instance.

# Importing the required packages
import numpy as np
import modern_robotics as mr

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics iterates in the body frame for an open chain robot
    and saves the joint vectors of each iterate in a .csv file.

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 4
    iterates = [thetalist]  # Storing the joint vectors of each iterate
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    print(" \n Iteration 1:")
    print("Joint vector:", thetalist)
    print("SE(3) end-effector config:")
    print(mr.TransInv(mr.FKinBody(M, Blist, thetalist)))
    print("Error twist V_b:", Vb)
    print("Angular error magnitude ||omega_b||:", np.linalg.norm([Vb[0], Vb[1], Vb[2]]))
    print("Linear error magnitude ||v_b||:", np.linalg.norm([Vb[3], Vb[4], Vb[5]]))
    print()

    while err and i < maxiterations:
        thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
        i = i + 1
        iterates.append(thetalist)  # Saving the joint vector of the current iterate
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

        print("Iteration", i + 1, ":") # Printing i + 1th value as the indexing starts from 0 in Python
        print("Joint vector:", thetalist)
        print("SE(3) end-effector config:")
        print(mr.TransInv(mr.FKinBody(M, Blist, thetalist)))
        print("Error twist V_b:", Vb)
        print("Angular error magnitude ||omega_b||:", np.linalg.norm([Vb[0], Vb[1], Vb[2]]))
        print("Linear error magnitude ||v_b||:", np.linalg.norm([Vb[3], Vb[4], Vb[5]]))
        print()

    # Saving the joint vectors as a .csv file with the name: iterates.csv
    iterates_matrix = np.array(iterates)
    np.savetxt('iterates.csv', iterates_matrix, delimiter=',')


# Testing input
W1,W2,L1,L2,H1,H2 = np.array([109,82,425,392,89,95])*1e-3
Blist = np.array([[0, 1, 0, W1+W2, 0, L1+L2],
                  [0, 0, 1, H2, -L1-L2, 0],
                  [0, 0, 1, H2, -L2, 0],
                  [0, 0, 1, H2, 0, 0],
                  [0,-1, 0, -W2, 0, 0],
                  [0, 0, 1, 0, 0, 0]]).T

M = np.array([[ -1, 0,  0, L1+L2],
              [ 0, 0,  1, W1+W2],
              [ 0, 1,  0, H1-H2],
              [ 0, 0,  0, 1]])

T = np.array([[0, 1, 0, -0.5],
              [0, 0, -1, 0.1],
              [-1, 0, 0, 0.1],
              [0, 0,  0, 1]])

thetalist0 = np.array([6.1,-2.4,4.5,-5.2,3.3,1.6])
# The mr.IKinBody() function was first run to obtain the output and then, values closer to the desired angles were used to get the iterates in this code.


# # Getting the theta values:

# W1,W2,L1,L2,H1,H2 = np.array([109,82,425,392,89,95])*1e-3
# Blist = np.array([[0, 1, 0, W1+W2, 0, L1+L2],
#                   [0, 0, 1, H2, -L1-L2, 0],
#                   [0, 0, 1, H2, -L2, 0],
#                   [0, 0, 1, H2, 0, 0],
#                   [0,-1, 0, -W2, 0, 0],
#                   [0, 0, 1, 0, 0, 0]]).T
# M = np.array([[ -1, 0,  0, L1+L2],
#               [ 0, 0,  1, W1+W2],
#               [ 0, 1,  0, H1-H2],
#               [ 0, 0,  0, 1]])
# T = np.array([[0, 1, 0, -0.5],
#               [0, 0, -1, 0.1],
#               [-1, 0, 0, 0.1],
#               [0, 0,  0, 1]])

# eomg = 0.001

# ev = 0.0001

# IKB = mr.IKinBody(Blist, M, T, thetalist0, eomg, ev)

# IKB

# Output: (array([ 6.14040452, -2.47950979,  4.54355065, -5.20563307,  3.28437176, 1.57079666]), True)


eomg = 0.001

ev = 0.0001

# Calling the function
result = IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)