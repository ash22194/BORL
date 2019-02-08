clear;
close all;
clc;

%% Parameters
m = 1; mass_factor = 2;
l = 1; length_factor = 1;
b = 0.5;
g = 9.81;
dt = 0.001;
numPointsx = 51;
numPointsx_dot = 51;
numPointsu = 21;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-5,5]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = 0.6*eye(2);
u_limits = [-5,5];
R = 1;
gtol = 0.001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 0.99;
maxiter = 30;
max_policy_iter = 30;
visualize = false;
test_policy = false;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

x = [x_limits(1), x_limits(2)];
y = [x_dot_limits(1), x_dot_limits(2)];
imagesc(x,y, V);
xlabel('theta'); ylabel('theta-dot');
title('Value Iteration');
colorbar;
pause(0.5);

%% Build GP based estimate of value function using the computed policy
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
[grid_x,grid_x_dot] = meshgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
p_ = @(s) policy(p, grid_x, grid_x_dot, s);
% nu = min(dx,dx_dot);
nu = 0;
sigma0 = 0;
sigmak = sqrt(1);
env = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
gptd = GPTD(env, nu, sigma0, sigmak, gamma_);
max_episode_length = 100;
number_of_episodes = 5000;
debug_ = true;
gptd.build_posterior_monte_carlo(p_, number_of_episodes, max_episode_length, debug_);
gptd.visualize(grid_x,grid_x_dot);
hold on;
scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
hold off;

function a = policy(p, grid_x, grid_x_dot, s)
  a = interp2(grid_x, grid_x_dot, p, s(1), s(2));
end

            