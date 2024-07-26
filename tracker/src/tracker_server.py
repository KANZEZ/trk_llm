#! /usr/bin/env python
"""
Script for running multi-agent tracking simulation and visualization
Also used for defining ROS setup (to be added)
"""

import os

import rospy
import rospkg

from tracker_manger import TrackerManager
from visualizer import Visualizer


class TrackerServer:

    def __init__(self, exp_name):
        """
        exp_name: .yaml config file to read
        """

        ######### general parameters #########
        ros_path = rospkg.RosPack().get_path('tracker')
        #ros_path = "/home/grasp/ma_ws/src/target_tracking/src/Rtracker"
        config_path = ros_path + "/config/" + exp_name + ".yaml"
        if not os.path.exists(config_path):
            print("Config file does not exist")
            return

        print("========= loading config file from: ", config_path)
        self.tracker_manager = TrackerManager(config_path)
        self.target_ids = self.tracker_manager.targetID
        self.robot_ids = self.tracker_manager.robotID
        self.dim = self.tracker_manager.dim
        self.save_path = ros_path + "/results/" + str(self.tracker_manager.testID) + "/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ######### ROS parameters #########
        self.target_odom_subs = []
        self.drone_odom_subs = []
        self.drone_cmd_pubs = []

        ############ visualization ############
        self.vis = Visualizer(self.save_path)
        self.his_target_pos = []
        self.his_target_vel = []
        self.his_drone_pos = []
        self.his_drone_vel = []
        self.comm_dzones = []
        self.sens_dzones = []
        self.his_drone_cmd = []

        # run simulation
        self.simulation()


    def simulation(self):
        """
        steps: int, number of steps to run the simulation
        """

        # load map
        print(" ========= loading map =========")
        self.vis.visualize_map(self.tracker_manager.x_bounds)
        self.vis.visualize_zones(self.tracker_manager.danger_zones)

        # solve the problem
        print(" ========= solving problem =========")
        reults = self.tracker_manager.solve_problem(self.tracker_manager.steps, self.vis, enable_animate=True)

        self.his_drone_pos = reults["robot_pos"]
        self.his_drone_vel = reults["robot_vel"]
        self.his_target_pos = reults["target_pos"]
        self.his_drone_cmd = reults["robot_cmd"]

        # visualize results
        print(" ========= visualising results =========")
        self.vis.visualize_target(self.his_target_pos, self.tracker_manager.targetID)
        self.vis.visualize_robot(self.his_drone_pos, self.tracker_manager.robotID)
        self.vis.vis_trace(self.tracker_manager.trace_list, self.tracker_manager.steps)
        self.vis.show()


if __name__ == '__main__':
    exp_name = "exp1"
    tracker_server = TrackerServer(exp_name)
    rospy.spin()
