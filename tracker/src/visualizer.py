import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from matplotlib.gridspec import GridSpec


class Visualizer:

    def __init__(self, save_path):
        self.save_path = save_path

        # covert into year-month-day-hour-minute-second
        self.plot_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.robot_marker = ['-c', '-g', '-k']
    

    def plot_dyn(self, robot_cmd, robot_vel, size = 1):

        #only on self.fig2
        plt.figure("Robot Dynamics")  # "Robot Dynamics

        num_robots = len(robot_cmd[0])
        cmd_label = ['cmd_x', 'cmd_y', 'cmd_z']
        vel_label = ['vel_x', 'vel_y', 'vel_z']
        
        #nrows, ncols, index
        for i in range(num_robots):
            plt.subplot(num_robots, 2, 1 + i)
            t = np.arange(0, len(robot_cmd), 1)
            x = [robot_cmd[j][i][0] for j in range(len(robot_cmd))]
            y = [robot_cmd[j][i][1] for j in range(len(robot_cmd))]
            
            plt.plot(t, x, label = cmd_label[0])
            plt.plot(t, y, label = cmd_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('cmd')
            plt.title('robot' + str(i))
        
        for i in range(num_robots):
            plt.subplot(num_robots, 2, 1 + num_robots + i)
            t = np.arange(0, len(robot_vel), 1)
            x = [robot_vel[j][i][0] for j in range(len(robot_vel))]
            y = [robot_vel[j][i][1] for j in range(len(robot_vel))]
            
            plt.plot(t, x, label = vel_label[0])
            plt.plot(t, y, label = vel_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('vel')

        plt.savefig(self.save_path + 'dyn_cmd_' + self.plot_name  + '.png')



    
    def plot_cmd(self, robot_cmd, size = 1):

        #only on self.fig2
        plt.figure("Robot Commands")  #

        #use a new figure
        print ("Visualizing robot command")
        ##plot cmd with time
        # use subplots to plot 3d position
        num_robots = len(robot_cmd[0])
        vis_label = ['cmd_x', 'cmd_y', 'cmd_z']


        for i in range(num_robots):
            plt.subplot(num_robots, 1, 1 + i)
            t = np.arange(0, len(robot_cmd), 1)
            x = [robot_cmd[j][i][0] for j in range(len(robot_cmd))]
            y = [robot_cmd[j][i][1] for j in range(len(robot_cmd))]
            
            plt.plot(t, x, label = vis_label[0])
            plt.plot(t, y, label = vis_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('cmd')
            plt.title('robot' + str(i))
        
        #fig2 = plt.figure(2)
        #fig is title: vel
        plt.savefig(self.save_path + 'cmd_' + self.plot_name  + '.png')
    

    def plot_vel(self, robot_vel, size = 1):

        #only on self.fig2
        plt.figure("Robot Velocity")  # "Robot Velocity
        # set size of the figure
        #plt.figure(figsize=(10, 10))

        #use a new figure
        print ("Visualizing robot velocity")
        ##plot cmd with time
        # use subplots to plot 3d position
        num_robots = len(robot_vel[0])
        vis_label = ['vel_x', 'vel_y', 'vel_z']
        print ("num_robots: ", num_robots)

        for i in range(num_robots):
            self.fig2 = plt.subplot(num_robots, 1, 1 + i)
            t = np.arange(0, len(robot_vel), 1)
            x = [robot_vel[j][i][0] for j in range(len(robot_vel))]
            y = [robot_vel[j][i][1] for j in range(len(robot_vel))]
            
            plt.plot(t, x, label = vis_label[0])
            plt.plot(t, y, label = vis_label[1])
            plt.legend()
            plt.grid(True)
            plt.ylabel('vel')
            plt.title('robot' + str(i))
        
        plt.savefig(self.save_path + 'vel_' + self.plot_name  + '.png')


    def plot_gradient_ellipse(self, pos, length, width, color = 'green'):
        plt.figure(1)
        plt.plot(pos[0], pos[1], 'o', color = 'black', markersize = 3)
        
        # Define a gradient colormap
        cmap_colors = [color, (1, 1, 1)]  # White to black gradient
        cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        
        # Define the grid
        x = np.linspace(pos[0] - length, pos[0] + length, 50)
        y = np.linspace(pos[1] - width, pos[1] + width, 50)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from center
        distances = np.sqrt((X - pos[0])**2 / length**2 + (Y - pos[1])**2 / width**2)
       

        # filter X, Y, distances which outside the ellipse
        # Create a mask for points outside the ellipse
        outside_ellipse = distances > 1

        # Mask out points outside the ellipse
        X[outside_ellipse] = np.nan
        Y[outside_ellipse] = np.nan
        distances[outside_ellipse] = np.nan
            

        # Plot the ellipse with gradient color. no boundaru
        plt.scatter(X, Y, c=distances, cmap=cmap, edgecolors='none')
        



    def visualize_map(self, x_bounds):
        plt.figure(1)
        ## self.fig size is large enough to contain the map
        ## plot the map, map can be 2d or 3d box
        print ("Visualizing map")
        map_dim = len(x_bounds)
        if map_dim == 2:
            plt.plot([x_bounds[0][0], x_bounds[0][1]], [x_bounds[1][0], x_bounds[1][0]], 'black')
            plt.plot([x_bounds[0][0], x_bounds[0][1]], [x_bounds[1][1], x_bounds[1][1]], 'black')
            plt.plot([x_bounds[0][0], x_bounds[0][0]], [x_bounds[1][0], x_bounds[1][1]], 'black')
            plt.plot([x_bounds[0][1], x_bounds[0][1]], [x_bounds[1][0], x_bounds[1][1]], 'black')

        plt.axis('tight')

        return


    def visualize_zones(self, zones):
        plt.figure(1)

        ## zones is circle with mu as center and conv as radius

        print ("Visualizing zones")
        for i in range(zones.nTypeI):
            #plot a disk
            # plot, fill with gradient color centered at mu
            # color is red 
            color = 'red'
            self.plot_gradient_ellipse(zones.typeI_mu[i], zones.typeI_cov[i][0], zones.typeI_cov[i][1], color)

        
        for i in range(zones.nTypeII):
            #plot a disk
            color = 'blue'
            self.plot_gradient_ellipse(zones.typeII_mu[i], zones.typeII_cov[i][0], zones.typeII_cov[i][1], color)


    def visualize_target(self, target_pos, target_ids, size = 1):
        plt.figure(1)

        print ("Visualizing target")

        ##plot id name as "tar" + str(i) at the start position
        for i in range(len(target_pos[0])):
            plt.text(target_pos[0][i][0] + 0.05 , target_pos[0][i][1] + 0.05 , 'tar' + str(target_ids[i]), fontsize=12)

            #plot the start position
            plt.plot(target_pos[0][i][0], target_pos[0][i][1], 'o', color = 'black', markersize = 5 * size)


        dim = len(target_pos[0])
        
        for i in range(len(target_pos)):
            color_ratio = i / len(target_pos)
            color = (color_ratio, 0, 1 - color_ratio)
            #print (target_pos[i])

            for j in range(len(target_pos[i])):
                ## for each target, plot the position
                plt.plot(target_pos[i][j][0], target_pos[i][j][1], 'o', color = color, markersize = 3*size)
                ## hold on 

            
        return
    

    def visualize_robot(self, robot_pos, robot_ids, size = 1):
        plt.figure(1)

        print ("Visualizing robot")
        ##plot id name as "rob" + str(i) at the start position
        for i in range(len(robot_pos[0])):
            plt.text(robot_pos[0][i][0] + 0.05 , robot_pos[0][i][1] + 0.05 , 'rob' + str(robot_ids[i]), fontsize=12)

            #plot the start position
            plt.plot(robot_pos[0][i][0], robot_pos[0][i][1], 'o', color = 'black', markersize = 5 * size)


        dim = len(robot_pos[0])
        
        for i in range(len(robot_pos)):
            color_ratio = i / len(robot_pos)
            # use different color for robots and targets
            color = (1 - color_ratio, color_ratio, 0)
            #print (robot_pos[i])

            for j in range(len(robot_pos[i])):
                ## for each robot, plot the position
                marker = '*'
                if j == 2:
                    marker = 'o'
                    color = (color_ratio, 1 - color_ratio, 0)
                plt.plot(robot_pos[i][j][0], robot_pos[i][j][1], marker, color = color, markersize = 2*size)
                ## hold on 

            
        return
    
    def show(self):
        #same length
        #plt.axis('equal')
        # use time as the name of the plot
        plt.figure(1)
        plt.savefig(self.save_path + 'target_' + self.plot_name  + '.png')
        plt.show()
        return


    def draw_ekf(self, estx, trux, esty, truy, bound):
        plt.figure(1)
        plt.scatter(estx, esty, marker='*')
        plt.scatter(trux, truy, marker='o')
        plt.xlim(bound[0])
        plt.ylim(bound[1])


    def create_animate(self, robot_pos, target_pos, robot_ids, target_ids, size = 1):
        """
        create a video
        robot_pos: nbot x dim
        target_pos: ntar x dim
        """
        # create a video
        fig = plt.figure(2)
        gs = GridSpec(1, 1, figure=fig)

        # Plot trajectory
        ax = fig.add_subplot(gs[:, 0])
        plt.xlim([-3.0, 3.0])
        plt.ylim([-3.0, 3.0])
        for i in range(robot_pos.shape[0]):
            ax.text(robot_pos[i, 0] + 0.05, robot_pos[i, 1] + 0.05, 'rob' + str(robot_ids[i]), fontsize=12)
            ax.plot(robot_pos[i, 0], robot_pos[i, 1], self.robot_marker[i], color='black', markersize=5 * size)

        for i in range(target_pos.shape[0]):
            ax.text(target_pos[i, 0] + 0.05, target_pos[i, 1] + 0.05, 'tar' + str(target_ids[i]), fontsize=12)
            ax.plot(target_pos[i, 0], target_pos[i, 1], '-b', color='black', markersize=5 * size)

        plt.tight_layout()



    def animate(self, robot_pos, target_pos, step, total_step , size = 1):
        """
        animate the video
        robot_pos: nbot x dim x nsteps, where nstep is the current sim step
        same as target_pos
        """
        fig = plt.gcf()  # Get current figure
        ax2 = fig.axes  # Get current axes of figure 2

        for i in range(target_pos.shape[0]):
            ax2[0].get_lines().pop(-1).remove()
        for i in range(robot_pos.shape[0]):
            ax2[0].get_lines().pop(-1).remove()

        for i in range(target_pos.shape[0]):
            ax2[0].plot(target_pos[i, 0, 0:step], target_pos[i, 1, 0:step], '-b', markersize=2*size)
        for i in range(robot_pos.shape[0]):
            ax2[0].plot(robot_pos[i, 0, 0:step], robot_pos[i, 1, 0:step], self.robot_marker[i], markersize=2 * size)


        plt.pause(0.03)

        plt.draw()


    def vis_trace(self, trace_list, total_step):
        plt.figure(3)
        plt.plot(np.arange(total_step), np.array(trace_list),'-o')







        