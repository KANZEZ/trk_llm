import numpy as np
import dynamics
import casadi


class Problem:

    def __init__(self, problem):
        """
        problem: dict, problem setup
        """
        self.robotID = problem['robotID']
        self.nRobot = len(self.robotID)
        self.targetID = problem['targetID']
        self.nTarget = len(self.targetID)
        self.targetStartPos = problem['targetStartPos']
        self.targetStartVel = problem['targetStartVel']
        self.robotStartPos = problem['robotStartPos']
        self.robotStartVel = problem['robotStartVel']

        self.dt = problem['dt']
        self.dim = problem['dim']
        self.N = problem['N']
        self.x_bounds = np.array(problem['x_bounds'])
        self.u_bounds = np.array(problem['u_bounds'])
        self.weights = problem['weights']
        self.task_assignment_matrix = np.array(problem['task_assignment_matrix'])
        self.target_dyn_type = problem['target_dyn_type']
        self.robot_dyn_type = problem['robot_dyn_type']
        self.min_dist = problem['min_dist']
        self.max_dist = problem['max_dist']
        ## parameters for the measurement of range and bearing sensors
        self.range_peak = problem['range_sensor'][0]
        self.range_shape = problem['range_sensor'][1]
        self.bearing_peak = problem['bearing_sensor'][0]
        self.bearing_shape = problem['bearing_sensor'][1]

        self.target_dyn_func = getattr(dynamics, self.target_dyn_type)
        self.robot_dyn_func = getattr(dynamics, self.robot_dyn_type)
        if self.target_dyn_type == 'second_order_dynamics':
            self.tstate_len = self.dim*2
        elif self.target_dyn_type == 'first_order_dynamics':
            self.tstate_len = self.dim
        if self.robot_dyn_type == 'second_order_dynamics':
            self.rstate_len = self.dim*2
        elif self.robot_dyn_type == 'first_order_dynamics':
            self.rstate_len = self.dim

        # position, velocity
        self.robot_odom = np.zeros((self.nRobot, self.rstate_len))
        self.target_odom = np.zeros((self.nTarget, self.tstate_len ))
        self.target_path_N = np.zeros((self.nTarget, self.dim, self.N)) #target real pos at time t+1, repeated N stages
            
        ##  information of targets
        # best estimation of target position at time t+1, repeated N stages
        self.targets_best_est_pos = np.zeros((self.nTarget, self.dim, self.N))
        self.targets_best_est_cov = np.zeros((self.nTarget, self.dim, self.dim, self.N))

        ### nessary dimension definition
        self.est_pos_idx = self.dim
        self.est_cov_idx = self.dim
        self.Lrvars = self.est_pos_idx + self.est_cov_idx # x y cov(00) cov(11) # len of running parameters
        self.Ldvars = self.dim * 2 # xv # len of decision var

        self.pos_idx = np.array([0, self.dim])  # start and end idx
        self.vel_idx = np.array([self.dim, self.dim * 2])  # start and end idx

        self.init_sim()

    
    def init_sim(self):
        """
        init problem setup parameters
        """
        ## initialize the robot and target odom
        self.robot_odom[:, :self.dim] = self.robotStartPos
        #self.robot_odom[:, self.dim:] = self.robotStartVel
        self.target_odom[:, :self.dim] = self.targetStartPos
        self.target_odom[:, self.dim:] = self.targetStartVel

        ## initialize the target path and target estimation
        for stage in range(self.N):
            self.targets_best_est_pos[:, :, stage] = self.targetStartPos
            self.targets_best_est_cov[:, :, :, stage] = np.eye(self.dim) * 0.2
            self.target_path_N[:, :, stage] = self.targetStartPos


    def sim_next_robot_i(self, cur_odom_i, cmd):
        nextodom = self.robot_dyn_func(cmd, cur_odom_i, self.dt)
        return nextodom

    def sim_next_target_i(self, cur_odom_i, cmd):
        nextodom = self.target_dyn_func(cmd, cur_odom_i, self.dt)
        return nextodom

    def sim_next_robot(self, cur_odom, cmd):
        """
        cur_odom: self.nRobot x self.dim * 2
        cmd : self.nRobot x self.dim
        """
        next_robot_odom = np.zeros((self.nRobot, self.rstate_len))
        for i in range(self.nRobot):
            next_robot_odom[i] = self.sim_next_robot_i(cur_odom[i], cmd[i])
        return next_robot_odom

    def sim_next_target(self, cur_odom):
        """
        cur_odom: self.nTarget x self.dim * 2
        """
        next_target_odom = np.zeros((self.nTarget, self.tstate_len))
        u_tar = np.zeros(self.dim) # constant velocity for target
        for i in range(self.nTarget):
            next_target_odom[i] = self.target_dyn_func(u_tar, cur_odom[i], self.dt)
        return next_target_odom

    def update_robot(self, cmd):
        """
        update_real_robot_odom
        cmd: self.nRobot x self.dim
        """
        self.robot_odom = self.sim_next_robot(self.robot_odom, cmd)
        return self.robot_odom

    def propagate_target_pos(self):
        """
        propagate the target real position for N stages, and update the target odom
        """
        next = self.sim_next_target(self.target_odom)
        pos_next = next[:, :self.dim]
        for stage in range(self.N):
            self.target_path_N[:, :, stage] = pos_next
        self.target_odom = next



    def get_trace(self, robot_pos_t):
        """
        get trace at the current time t
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        trace = 0
        target_est_pos_t = self.targets_best_est_pos[:, :, 0]
        target_est_cov_t = self.targets_best_est_cov[:, :, :, 0]

        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            tar_est_pos_k = target_est_pos_t[tar, :].T
            tar_est_cov_k = target_est_cov_t[tar, :]

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            # for the observation by all robots
            H_k1 = np.zeros((self.dim * 1, self.dim))
            R_k1 = []

            for rob in range(self.nRobot):
                if self.task_assignment_matrix[rob, tar] == 0:
                    continue
                rob_pos_k = robot_pos_t[rob, :self.dim].T
                p_tilde = rob_pos_k - tar_est_pos_pred

                # get ranging and bearing sensors' state vector
                h_range = -1 * p_tilde / np.linalg.norm(p_tilde)
                h_bearing = rot @ p_tilde / np.dot(p_tilde, p_tilde)
                # measurement matrix for rob at time k
                H_rob_k1 = np.array([h_range, h_bearing])
                H_k1[0 * self.dim: (0 + 1) * self.dim, :] = H_rob_k1

                # get measurement error and R for rob at time k
                dist_k1 = np.linalg.norm(rob_pos_k - tar_est_pos_pred)
                R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1)
                R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1)
                R_k1.append(1 / R_range_inv)
                R_k1.append(1 / R_bearing_inv)

            # measurement update for this target:
            R = np.diag(R_k1)

            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            trace += np.trace(tar_est_cov_k1)

        return trace



    def ekf_update(self, robot_pos_t):
        """
        EKF update the best estimation of targets at time t+1
        robot_pos_t: self.nRobot x self.dim, pos at current time t
        """
        A = np.eye(self.dim)
        Q = 0.05 * np.eye(self.dim)
        rot = np.array([[0, 1], [-1, 0]])

        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            # get the current estimation of tar at time k
            tar_est_pos_k = self.targets_best_est_pos[tar, :, 1].T
            tar_est_cov_k = self.targets_best_est_cov[tar, :, :, 1].T

            # prediction step of EKF:
            tar_est_pos_pred = A @ tar_est_pos_k
            tar_est_cov_pred = A @ tar_est_cov_k @ A.T + Q

            # truth position for target at k+1
            tar_truth_pos = self.target_path_N[tar, :, 0]

            # for the observation by all robots
            H_k1 = np.zeros((self.dim * 1, self.dim))
            R_k1 = []
            z_tildes = np.zeros(self.dim * 1)

            for rob in range(self.nRobot):
                if self.task_assignment_matrix[rob, tar] == 0:
                    continue

                # diff position between robot and target prediction
                rob_pos_k1 = robot_pos_t[rob, :self.dim]
                p_tilde = rob_pos_k1 - tar_est_pos_pred

                # get ranging and bearing sensors' state vector
                h_range = -1 * p_tilde / np.linalg.norm(p_tilde)
                h_bearing = rot @ p_tilde / np.dot(p_tilde, p_tilde)

                # measurement matrix for rob at time k+1
                H_rob_k1 = np.array([h_range, h_bearing])
                H_k1[0 * self.dim: (0 + 1) * self.dim, :] = H_rob_k1

                # get measurement error and R for rob at time k+1
                # generate the noise for the measurement
                dist_k1 = np.linalg.norm(rob_pos_k1 - tar_truth_pos)
                R_range_inv = self.range_peak * np.exp(-self.range_shape * dist_k1)
                R_bearing_inv = self.bearing_peak * np.exp(-self.bearing_shape * dist_k1)
                R_k1.append(1 / R_range_inv)
                R_k1.append(1 / R_bearing_inv)

                range_noise = np.random.normal(0, 1 / R_range_inv)
                bearing_noise = np.random.normal(0, 1 / R_bearing_inv)

                z_tilde_rob = H_rob_k1 @ (tar_truth_pos - tar_est_pos_pred) + np.array([range_noise, bearing_noise])
                z_tildes[0 * self.dim: (0 + 1) * self.dim] = z_tilde_rob

            # measurement update for this target:
            R = np.diag(R_k1)
            S = H_k1 @ tar_est_cov_pred @ H_k1.T + R
            K = tar_est_cov_pred @ H_k1.T @ np.linalg.inv(S)
            tar_est_pos_k1 = tar_est_pos_pred + K @ z_tildes
            tar_est_cov_k1 = tar_est_cov_pred - K @ S @ K.T

            # update the best estimation for time k+1
            for iStage in range(self.N):
                self.targets_best_est_pos[tar, :, iStage] = tar_est_pos_k1.T
                self.targets_best_est_cov[tar, :, :, iStage] = tar_est_cov_k1


    def trace_objective_casadi(self, z, p):
        """
        get trace value, casadi version
        last stage objective function at each iteration
        @param:  z: decision vars: robot [x, y, vx, vy, ax, ay, robot2:...] at last stage
        @param:  p: running parameters at last stage:
        [target1 best_estimation_pos at k+1(x, y),
        target1 best_estimation_covat k+1(cov(0,0), cov(1,1)), target2 ...]
        @return: trace
        """
        trace = 0
        A = casadi.SX.eye(self.dim)
        Q = 0.05 * casadi.SX.eye(self.dim)
        rot = casadi.SX(np.array([[0, 1], [-1, 0]]))


        # for every target, we get its best estimation by the observations of all robots
        for tar in range(self.nTarget):
            # get the best estimation of tar at time k+1
            tar_est_pos_k = p[tar * self.Lrvars: tar * self.Lrvars + self.est_pos_idx]
            tar_est_cov_k = casadi.diag(p[tar * self.Lrvars + self.est_pos_idx: (tar + 1) * self.Lrvars])

            # prediction step of EKF:
            tar_est_pos_pred = casadi.mtimes(A, tar_est_pos_k)
            tar_est_cov_pred = casadi.mtimes(casadi.mtimes(A, tar_est_cov_k),  A.T) + Q

            # for the observation by all robots
            H_k1 = casadi.SX.zeros((self.dim * 1, self.dim))
            R_k1 = []

            for rob in range(self.nRobot):
                if self.task_assignment_matrix[rob, tar] == 0:
                    continue

                rob_pos_k1 = z[rob * self.Ldvars: rob * self.Ldvars + self.dim]
                p_tilde = rob_pos_k1 - tar_est_pos_pred

                # get ranging and bearing sensors' state vector
                h_range = -1 * p_tilde / casadi.norm_2(p_tilde)
                h_bearing = casadi.mtimes(rot, p_tilde) / casadi.mtimes(p_tilde.T, p_tilde)

                # measurement matrix for rob at time k+1
                H_rob_k1 = casadi.horzcat(h_range, h_bearing)
                H_k1[0 * self.dim: (0 + 1) * self.dim, :] = H_rob_k1
                # get measurement error and R for rob at time k+1
                # generate the noise for the measurement
                dist_k1 = casadi.norm_2(p_tilde)
                R_range_inv = self.range_peak * casadi.exp(-self.range_shape * dist_k1)
                R_bearing_inv = self.bearing_peak * casadi.exp(-self.bearing_shape * dist_k1)
                R_k1.append(1 / R_range_inv)
                R_k1.append(1 / R_bearing_inv)

            # measurement update for this target:
            R_k1 = casadi.vertcat(*R_k1)
            R = casadi.diag(R_k1)

            S = casadi.mtimes(casadi.mtimes(H_k1, tar_est_cov_pred), H_k1.T) + R
            K = casadi.mtimes(casadi.mtimes(tar_est_cov_pred, H_k1.T), casadi.inv(S))

            tar_est_cov_k1 = tar_est_cov_pred - casadi.mtimes(casadi.mtimes(K, S), K.T)

            trace += casadi.trace(tar_est_cov_k1)


        return trace



