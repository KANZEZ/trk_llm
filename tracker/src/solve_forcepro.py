"""
this script is for the generation of forcespro solver and model
"""
import casadi
import numpy as np
import forcespro
import forcespro.nlp
import dynamics

class ForcesProSolver:

    def __init__(self, problem, zones):
        """
        problem: the problem class
        zones: the danger zones class
        """
        self.problem = problem
        self.dt = self.problem.dt
        self.weights = self.problem.weights
        self.x_bounds = self.problem.x_bounds
        self.u_bounds = self.problem.u_bounds
        self.N = self.problem.N
        self.dt = self.problem.dt
        self.nRobot = self.problem.nRobot
        self.nTarget = self.problem.nTarget
        self.dim = self.problem.dim
        self.min_dist = self.problem.min_dist
        self.max_dist = self.problem.max_dist

        # dimension defination
        self.Ldvars = self.problem.Ldvars # decision var len at one stage for one robot
        self.Ldvars_nbot = self.problem.Ldvars * self.nRobot  # decision var len at one state for all robots
        self.LdvarsN = self.Ldvars_nbot * self.N  # decision var len at N stage

        self.Lrvars = self.problem.Lrvars  # running param len at one stage for one tar
        self.Lrvars_ntar = self.Lrvars * self.nTarget  # running param len at one state for all tar
        self.LrvarsN = self.Lrvars_ntar * self.N  # running param len at N stage

        self.pos_idx = self.problem.pos_idx
        self.vel_idx = self.problem.vel_idx
        self.input_idx = self.vel_idx
        self.input_dim = self.dim

        self.zones = zones


    def get_pos_var(self, z):
        """
        get the position var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_pos = casadi.SX.zeros(self.dim * self.nRobot)
        for i in range(self.nRobot):
            start_idx = i * self.Ldvars
            z_pos[i * self.dim: (i + 1) * self.dim] = \
                z[start_idx + self.pos_idx[0]: start_idx + self.pos_idx[1]]
        return z_pos


    def get_pos_idx(self):
        """
        get the position index list in desicion var z
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        xinitidx = []
        for i in range(self.nRobot):
            start_idx = i * self.Ldvars
            range_idx = range(start_idx + self.pos_idx[0], start_idx + self.pos_idx[1])
            xinitidx += range_idx
        return xinitidx


    def get_input_var(self, z):
        """
        get the input var in desicion var
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = casadi.SX.zeros(self.input_dim * self.nRobot)
        for i in range(self.nRobot):
            start_idx = i * self.Ldvars
            z_input[i * self.input_dim: (i + 1) * self.input_dim] = \
                z[start_idx + self.input_idx[0]: start_idx + self.input_idx[1]]
        return z_input


    def obj(self, z):
        """
        control input penalty
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        return self.weights[0] * casadi.sum1(z_input ** 2)


    def objN(self, z, p):
        """
        last stage trace at each iteration
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        p = running param: [x1, y1, cov11, cov22, x2, y2, cov11, cov22, ...]
        """
        return self.weights[1] * self.problem.trace_objective_casadi(z, p)


    def objN_dist(self, z, p):
        """
        last stage distance objective, not used
        """
        return 20 * ((z[0] - p[0])**2 + (z[1] - p[1])**2 + (z[4] - p[4])**2 + (z[5] - p[5])**2)


    def chance_constraints_typeI(self, z):
        """
        chance constraints for type I
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_pos = self.get_pos_var(z)
        values = casadi.SX.zeros((self.nRobot, self.zones.nTypeI))
        for iRobot in range(self.nRobot):
            for iZone in range(self.zones.nTypeI):
                robot_pos = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]
                values[iRobot, iZone] = self.zones.typeI_zone_value_i_casadi(robot_pos, iZone)
        value_vec = casadi.vec(values)
        return value_vec


    def collision_avoid_constraint(self, z):
        """
        collision avoidance between robots, casadi version
        @param:  z: decision vars: robot [x, y, vx, vy, x1, y1, vx1, vy1 ... ]
        @param:  p: running parameters at last stage:
        @return: constraints
        """
        z_pos = self.get_pos_var(z)
        values = []
        for iRobot in range(self.nRobot - 1):
            pos_i = z_pos[iRobot * self.dim: (iRobot + 1) * self.dim]
            for jRobot in range(iRobot + 1, self.nRobot):
                pos_j = z_pos[jRobot * self.dim: (jRobot + 1) * self.dim]
                dist_sqr = casadi.sumsqr(pos_i - pos_j)    # distance square, min^2 <= dist_sqr <= max^2
                values.append(dist_sqr)
        values_vec = casadi.vertcat(*[casadi.SX(x) for x in values])
        return values_vec


    def eq_constraint(self, z):
        """
        equality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        z_input = self.get_input_var(z)
        z_pos = self.get_pos_var(z)
        return dynamics.first_order_dynamics(z_input, z_pos, self.dt)


    def ineq_constraint(self, z):
        """
        inequality constraints
        z = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        return casadi.vertcat(self.chance_constraints_typeI(z),
                              self.collision_avoid_constraint(z))


    def get_E_matrix(self, row, col):
        """
        get the E matrix for equality constraints, on the LHS, E is num_eq x num_vars,
        E is selection matrix, select the position states from decision vars [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        E = np.zeros((row, col))
        position_idx, r = 0, 0
        for i in range(self.nRobot):
            E[r, position_idx] = 1
            E[r + 1, position_idx + 1] = 1
            position_idx += self.Ldvars
            r += self.dim
        return E



    def generate_solver(self):

        model = forcespro.nlp.SymbolicModel()
        model.N = self.problem.N
        model.nvar = self.Ldvars * self.nRobot  # [x1 y1 vx1 vy1, ...]

        num_collision_avoid_con = self.nRobot * (self.nRobot - 1) // 2
        num_chance_con = self.zones.nTypeI * self.nRobot
        model.nh = num_chance_con + num_collision_avoid_con  # number of chance constraints

        model.neq = self.dim * self.nRobot  # [x1, y1, x2, y2, ...]
        model.npar = self.Lrvars * self.nTarget  # [x1, y1, cov11, cov22, x2, y2, cov11', cov22' ...]

        # obj
        model.objective = self.obj
        model.objectiveN = self.objN

        model.eq = lambda z: self.eq_constraint(z)
        model.E = self.get_E_matrix(model.neq, model.nvar)

        # ineq
        model.ineq = lambda z: self.ineq_constraint(z)
        model.hl = np.concatenate(( np.zeros(num_chance_con),
                              self.min_dist**2 * np.ones(num_collision_avoid_con) ))
        model.hu = np.concatenate(( np.inf * np.ones(num_chance_con),
                             self.max_dist**2 * np.ones(num_collision_avoid_con) ))

        # bounds for decision variables
        bound_l = np.vstack((np.array([self.x_bounds[0, 0], self.x_bounds[1, 0]]),
                             np.array([self.u_bounds[0], self.u_bounds[0]]))).reshape(-1)
        bound_l = np.tile(bound_l, self.nRobot).reshape(-1)
        bound_u = np.vstack((np.array([self.x_bounds[0, 1], self.x_bounds[1, 1]]),
                             np.array([self.u_bounds[1], self.u_bounds[1]]))).reshape(-1)
        bound_u = np.tile(bound_u, self.nRobot).reshape(-1)

        model.lb = bound_l
        model.ub = bound_u

        # Initial condition idx
        model.xinitidx = self.get_pos_idx()

        # solver parameters defined
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 500  # Maximum number of iterations
        codeoptions.printlevel = 1  # Use printlevel = 2 to print progress (but
        #                             not for timings)
        codeoptions.optlevel = 0  # 0 no optimization, 1 optimize for size,
        #                             2 optimize for speed, 3 optimize for size & speed
        codeoptions.overwrite = 1
        codeoptions.cleanup = False
        codeoptions.timing = 1

        codeoptions.noVariableElimination = 1
        codeoptions.nlp.TolStat = 1E-3
        codeoptions.nlp.TolEq = 1E-3
        codeoptions.nlp.TolIneq = 1E-3
        codeoptions.nlp.TolComp = 1E-3

        # Creates code for symbolic model formulation given above, then contacts
        # server to generate new solver
        solver = model.generate_solver(options=codeoptions)

        return model, solver




