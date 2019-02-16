clear;
close all;
clc;

%% Parameters
m = 1;
l = 1;
b = 0.5;
g = 1;
dt = 0.01;
numPointsx = 51;
numPointsx_dot = 31;
numPointsu = 61;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(2) has goal x

x_dot_limits = [-2.2,2.2]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = 10*eye(2);
u_limits = [-10,10];
R = 0.01;
gtol = 0.001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 0.99;
maxiter = 30;
max_policy_iter = 30;
visualize = false;
test_policy = true;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

valueFig = figure();
subplot(2,1,1);
x = [x_limits(1), x_limits(2)];
y = [x_dot_limits(1), x_dot_limits(2)];
imagesc(x,y,V');
xlabel('theta'); ylabel('theta-dot');
title('Value Iteration');
colorbar;
pause(0.5);

%% Build GP based estimate of value function using the computed policy
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
du  = (u_limits(2)-u_limits(1))/(numPointsu-1);
[grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
% p_ = @(s) policy(p, grid_x, grid_x_dot, s);
% nu = min(dx,dx_dot);
nu = exp(-3/2);
sigma0 = 0.02;
epsilon = 0.1;
epsilon_end = 0.05;
is_reward = false;
discra = false;
env = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
sigma_s = [dx; dx_dot];
sigma_a = [du];
% sigma_s = [1;1];
% sigma_a = [1];
gpsarsa = GPSARSA(env, u_limits, discra, epsilon, nu, sigma_s, sigma_a, sigma0, gamma_, is_reward);
max_episode_length = 100;
number_of_episodes = 150;
debug_ = true;
states = [reshape(grid_x,size(grid_x,1)*size(grid_x,2),1), reshape(grid_x_dot,size(grid_x_dot,1)*size(grid_x_dot,2),1)]';

gpsarsa.build_policy_monte_carlo_fixed_starts(states, number_of_episodes, max_episode_length, epsilon_end, debug_);
V = gpsarsa.get_value_function(states);

set(0,'CurrentFigure',valueFig);
subplot(2,1,2);
imagesc(reshape(V,numPointsx,numPointsx_dot));
colorbar;
% gptd.get_value_function(grid_x,grid_x_dot);
% hold on;
% scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
% hold off;
% function a = policy(p, grid_x, grid_x_dot, s)
%   a = interp2(grid_x, grid_x_dot, p, s(1), s(2));
% end

            