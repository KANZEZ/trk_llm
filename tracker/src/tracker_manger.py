import numpy as np
import yaml
from problem import Problem
from danger_zones import DangerZones
from solve_forcepro import ForcesProSolver

import sys


class TrackerManager:

    def __init__(self, config_path):
        """
        config_path: .yaml config file to read
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        ############# Problem #############
        self.testID = self.config['testID']
        self.steps = self.config['steps']

        self.problem = self.config['Problem']
        self.robotID = self.problem['robotID']
        self.nRobot = len(self.robotID)
        self.targetID = self.problem['targetID']
        self.nTarget = len(self.targetID)
        self.x_bounds = self.problem['x_bounds']
        self.dt = self.problem['dt']
        self.dim = self.problem['dim']
        self.N = self.problem['N'] # horizon(stages)

        self.trac_prob = Problem(self.problem)

        ############# Danger Zone #############
        self.Zones = self.config['Zones']
        self.danger_zones = DangerZones(self.Zones, self.dim)

        ############# Solver #############
        forcespro = ForcesProSolver(self.trac_prob, self.danger_zones)
        self.model, self.solver = forcespro.generate_solver()

        self.state = np.zeros((self.nRobot, self.dim))
        self.cmd = np.zeros((self.nRobot, self.dim))

        # for animate and visualization
        self.accumulate_state_rob = np.zeros((self.nRobot, self.dim, self.steps+1))
        self.accumulate_state_tar = np.zeros((self.nTarget, self.dim, self.steps+1))
        self.trace_list = []

        self.last_exitflag = 1


    def update_running_params(self, problem_setup):
        """
        update new best est pos and voc for targets
        """
        next_best_est_tar_pos = np.copy(self.trac_prob.targets_best_est_pos)
        next_best_est_tar_cov = np.copy(self.trac_prob.targets_best_est_cov)
        order = []
        for iRobot in range(self.nRobot):
            for iTar in range(self.nTarget):
                if self.trac_prob.task_assignment_matrix[iRobot, iTar] == 1:
                    order.append(iTar)
        # sort the order of the target at the first axis of next_best_est_tar_pos
        next_best_est_tar_pos = next_best_est_tar_pos[order, :, :]
        next_best_est_tar_cov = next_best_est_tar_cov[order, :, :, :]


        running_params = np.zeros(((self.dim + self.dim) * self.nTarget * self.N, 1))
        for iStage in range(self.N):
            for jTarget in range(self.nTarget):
                start_idx = iStage * (self.dim + self.dim) * self.nTarget + jTarget * (self.dim + self.dim)
                running_params[start_idx: start_idx + self.dim, 0] = next_best_est_tar_pos[jTarget, :,  iStage].T
                running_params[start_idx + self.dim: start_idx + (self.dim + self.dim), 0] = \
                    np.diag(next_best_est_tar_cov[jTarget, :, :, iStage])

        problem_setup["all_parameters"] = running_params


    """
    bug: when typeI_mu = [0.0, 0.0] and vx, vy in the initial guess= [0.0, 0.0], will have NAN or INF value?????
         has something to do with solver?
    """
    def update_init_guess(self, problem_setup):
        x0i = np.zeros((self.dim * 2 * self.nRobot, 1))
        xinit = np.zeros((self.dim * 1 * self.nRobot, 1))
        for iRobot in range(self.nRobot):
            start_idx_x0 = iRobot * self.dim * 2
            start_idx_init = iRobot * self.dim

            x0i[start_idx_x0: start_idx_x0 + 1 * self.dim, 0] = self.trac_prob.robot_odom[iRobot, :self.dim].T # x, y, vx=cmd, vy=cmd
            #### better to keep the initial guess of output ~ 0
            # if self.last_exitflag == 1: # if can solve normally
            #     x0i[start_idx_x0 + 1 * self.dim: start_idx_x0 + 2 * self.dim, 0] = self.cmd[iRobot, :].T
            x0i[start_idx_x0 + 1 * self.dim: start_idx_x0 + 2 * self.dim, 0] += 1e-10  # avoid NAN or INF value

            xinit[start_idx_init: start_idx_init + 1 * self.dim, 0] = self.trac_prob.robot_odom[iRobot, :self.dim].T  # only x, y

        x0 = np.transpose(np.tile(x0i, (1, self.N)))
        problem_setup["x0"] = x0
        problem_setup["xinit"] = xinit


    def extract_output(self, output):
        """
        get the first stage output
        """
        # Extract output
        temp = np.zeros((2*self.dim*self.nRobot, self.N))
        for i in range(0, self.N):
            temp[:, i] = output['x{}'.format(i + 1)]

        first_stage = 0
        cmd = np.zeros((self.nRobot, self.dim))
        state = np.zeros((self.nRobot, self.dim))
        for iRobot in range(self.nRobot):
            start_idx = iRobot * self.dim * 2
            state[iRobot, :] = temp[start_idx: start_idx + 1 * self.dim, first_stage].T
            cmd[iRobot, :] = temp[start_idx + 1 * self.dim: start_idx + 2 * self.dim, first_stage].T

        return state, cmd


    def step(self):
        """
        step the simulation for one iteration at time t
        """
        print("==================== step start ====================")
        # get target real state(propagate) at t + 1
        # filled N stages -> target_path_N
        self.trac_prob.propagate_target_pos()
        # compute trace at t (robot <-> best_est_tar_pos)
        cur_trace = self.trac_prob.get_trace(self.state)
        self.trace_list.append(cur_trace)

        # ekf update the best_est_tar_pos at time k+1
        self.trac_prob.ekf_update(self.state)

        # update running parameters(using target est k+1 info) for each iteration
        problem_setup = {}
        self.update_init_guess(problem_setup)
        self.update_running_params(problem_setup)
        # solve(objN(using target est k+1 info and decision var z to get trace at time t+1),
        # output includes next state of robot) and record data
        output, exitflag, info = self.solver.solve(problem_setup)

        print("exitflag = ", exitflag)
        #assert exitflag == 1 or exitflag == 0, "bad exitflag"
        self.last_exitflag = exitflag
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n" \
                         .format(info.it, info.solvetime))

        # update robot real state at t+1
        state, cmd = self.extract_output(output)
        self.state[:, :] = state[:, :]
        self.cmd[:, :] = cmd[:, :]
        self.trac_prob.update_robot(cmd)

        return cmd


    def solve_problem(self, sim_steps, visualizer, enable_animate=False):
        """
        for loop iteration to solve the problem
        """
        step = 0
        reults = {"robot_pos": [], "robot_vel": [], "target_pos": [], "robot_cmd": []}

        self.state[:, :] = self.trac_prob.robot_odom[:, :]
        self.accumulate_state_rob[:, :, step] = self.trac_prob.robot_odom[:, :]
        self.accumulate_state_tar[:, :, step] = self.trac_prob.target_odom[:, :self.dim]

        ### for animate
        if enable_animate:
            visualizer.create_animate(self.trac_prob.robot_odom[:, :],
                               self.trac_prob.target_odom[:, :self.dim], self.robotID, self.targetID)

        while step < sim_steps:
            #### for test
            # if step == 180:
            #     self.trac_prob.task_assignment_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            #     print("========================")
            cmd = self.step()

            # animate show
            if enable_animate:
                visualizer.animate(self.accumulate_state_rob, self.accumulate_state_tar, step, sim_steps)

            # record data
            reults["robot_pos"].append(self.trac_prob.robot_odom[:, :self.dim])
            reults["robot_vel"].append(self.trac_prob.robot_odom[:, self.dim:])
            reults["target_pos"].append(self.trac_prob.target_odom[:, :self.dim])
            reults["robot_cmd"].append(cmd)

            step += 1

            # update animate state
            self.accumulate_state_rob[:, :, step] = self.trac_prob.robot_odom[:, :]
            self.accumulate_state_tar[:, :, step] = self.trac_prob.target_odom[:, :self.dim]


        return reults

