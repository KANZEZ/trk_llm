import numpy as np
import scipy
import casadi
import scipy.integrate

class DangerZones:

    def __init__(self, zones, dim):

        self.nTypeI = zones['nTypeI']  ##### sensor zones
        self.nTypeII = zones['nTypeII'] ##### communication zones
        self.typeI_mu = zones['typeI_mu']
        self.typeI_mu = np.array(self.typeI_mu)
        self.typeI_cov = zones['typeI_cov']
        self.typeII_mu = zones['typeII_mu']
        self.typeII_cov = zones['typeII_cov']
        self.typeI_d = zones['typeI_d']  ## radius of sensor zones
        self.typeII_d = zones['typeII_d']
        self.delta = zones['delta']
        self.eps1 = zones['eps1']
        self.eps2 = zones['eps2']

        self.dim = dim


    def if_in_typeI_zones(self, robot_pos):

        flags = np.zeros((len(robot_pos), self.nTypeI))

        for iRobot in range(len(robot_pos)):
            for iZone in range(self.nTypeI):
                drone_dist = np.linalg.norm(robot_pos[iRobot][0:self.dim] - self.typeI_mu[iZone])
                if drone_dist < 0.85*self.typeI_d and flags[iRobot][iZone] == 0:
                    flags[iRobot][iZone] = 1
                elif drone_dist > self.typeI_d and flags[iRobot][iZone] == 1:
                    flags[iRobot][iZone] = 2
        
        return flags
    

    def if_in_typeII_zones(self, robot_pos):

        flags = np.zeros((len(robot_pos), self.nTypeII))

        for iRobot in range(len(robot_pos)):
            for iZone in range(self.nTypeII):
                drone_dist = np.linalg.norm(robot_pos[iRobot][0:self.dim] - self.typeII_mu[iZone])
                if drone_dist < 0.85*self.typeII_d and flags[iRobot][iZone] == 0:
                    flags[iRobot][iZone] = 1
                elif drone_dist > self.typeII_d and flags[iRobot][iZone] == 1:
                    flags[iRobot][iZone] = 2
        
        return flags


    '''
    @description: chance constraint for type I danger zones
    '''
    def typeI_zone_value_i_casadi(self, robot_pos, zone_idx):
        """
        robot_pos: decision var [x, y] for one robot
        """
        typeI_cov_matrix = np.diag(self.typeI_cov[zone_idx])
        ai = self.typeI_mu[zone_idx] - robot_pos
        ai_normalized = ai / casadi.norm_2(ai)
        erf_value = scipy.special.erfinv(1 - 2 * self.eps1)
        sqr_term = 2 * casadi.mtimes(ai_normalized.T, casadi.mtimes(typeI_cov_matrix, ai_normalized))
        cons = casadi.mtimes(ai_normalized.T, ai) - erf_value * casadi.sqrt(sqr_term) - self.typeI_d[zone_idx]   # >= 0 is ok
        value = cons

        return value

    '''
    @description: chance constraint for type I danger zones
    '''
    def typeII_zone_value_i_casadi(self, robot_pos, zone_idx, all_robot_pos, num_bot):
        """
        robot_i: current robot i pos decision var
        zone_idx: the index of the current zone
        all_robot_pos: the position of all robots
        num_bot: the number of robots
        """
        typeI_cov_matrix = np.diag(self.typeI_cov[zone_idx])
        ai = self.typeII_mu[zone_idx] - robot_pos
        ai_normalized = ai / casadi.norm_2(ai)
        erf_value = scipy.special.erfinv(1 - 2 * self.eps2)
        sqr_term = 2 * casadi.mtimes(ai_normalized.T, casadi.mtimes(typeI_cov_matrix, ai_normalized))

        max_neighbor_dist = -1
        for jRobot in range(num_bot):
            diff = robot_pos - all_robot_pos[jRobot, 0:self.dim]
            neighbor_dist = casadi.norm_2(diff)
            max_neighbor_dist = casadi.fmax(max_neighbor_dist, neighbor_dist)

        if max_neighbor_dist == -1:
            print("Error: max_neighbor_dist is -1")

        cons = casadi.mtimes(ai_normalized.T, ai) - erf_value * casadi.sqrt(sqr_term) - \
               self.delta[zone_idx] * max_neighbor_dist
        value = cons
        return value