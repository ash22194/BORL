clear;
close all;
clc;

%% Parameters
m = 1; mass_factor = 2;
l = 1; length_factor = 1;
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
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-2.2,2.2]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = 10*eye(2);
u_limits = [-10,10];
R = 0.01;
gtol = 0.001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 0.99;
maxiter = 300;
max_policy_iter = 30;
visualize = false;
test_policy = false;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(m*mass_factor, l*length_factor, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

valueFig = figure();
subplot(4,1,1);
x = [x_limits(1), x_limits(2)];
y = [x_dot_limits(1), x_dot_limits(2)];
imagesc(x, y, V');
xlabel('theta'); ylabel('theta-dot');
title('Target');
colorbar;
pause(0.5);

%% Compute value function for bootstrapping
[p_bootstrapped, V_bootstrapped] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

set(0,'CurrentFigure',valueFig);
subplot(4,1,2);
imagesc(x, y, V_bootstrapped');
xlabel('theta'); ylabel('theta-dot');
title('Starting point');
colorbar;
pause(0.5);

%% Build GP based estimate of value function using the computed policy
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
[grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
p_target = @(s) policy(p, grid_x, grid_x_dot, s);

env = pendulum(m*mass_factor, l*length_factor, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
dynamics = @(s,a)env.dynamics(s,a);
reward = @(s_,a)env.cost(s_,a);
states = [reshape(grid_x,size(grid_x,1)*size(grid_x,2),1), reshape(grid_x_dot,size(grid_x_dot,1)*size(grid_x_dot,2),1)];

V_interpolant = @(s) interpn(grid_x, grid_x_dot, reshape(V_bootstrapped, numPointsx, numPointsx_dot), s(1,:)', s(2,:)');

% nu = min(dx,dx_dot);
nu = exp(-1)-0.15;
sigma0 = 0.02;
sigmak = min(dx,dx_dot);
gptd_b = GPTD_bootstrapped(env, V_interpolant, nu, sigma0, sigmak, gamma_);
max_episode_length = 100;
number_of_episodes = 1500;
debug_ = true;
gptd_b.build_posterior_monte_carlo_fixed_starts(p_target, states', number_of_episodes, max_episode_length, debug_);
V_b = gptd_b.get_value_function(states');

set(0,'CurrentFigure',valueFig);
subplot(4,1,3);
imagesc(x, y, reshape(V_b,size(grid_x,1),size(grid_x_dot,2))');
xlabel('theta'); ylabel('theta-dot');
title('Bootstrapped GPTD');
colorbar;

%% Vanilla GPTD

gptd = GPTD_fast(env, nu, sigma0, sigmak, gamma_);
gptd.build_posterior_monte_carlo_fixed_starts(p_target, states', number_of_episodes, max_episode_length, debug_);
V_gptd = gptd.get_value_function(states');

set(0,'CurrentFigure',valueFig);
subplot(4,1,4);
imagesc(x, y, reshape(V_gptd,size(grid_x,1),size(grid_x_dot,2))');
xlabel('theta'); ylabel('theta-dot');
title('Vanilla GPTD');
colorbar;

function a = policy(p, grid_x, grid_x_dot, s)
  a = interpn(grid_x, grid_x_dot, p, s(1), s(2));
end
            