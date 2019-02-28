clear;
close all;
clc;

%% Parameters
m = 1; mass_factor = 2;
l = 1; length_factor = 1;
b = 0.5;
g = 9.81;
dt = 0.001;
numPointsx = 81;
numPointsx_dot = 101;
numPointsu = 51;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-3,3]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = [25,0;0,0.01];
u_limits = [-7.9,7.9];
R = 0.01;
gtol = 0.0001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 0.99;
maxiter = 500;
max_policy_iter = 30;
visualize = false;
test_policy = true;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

x = [x_limits(1), x_limits(2)];
y = [x_dot_limits(1), x_dot_limits(2)];
figure;
imagesc(x,y, V');
xlabel('theta'); ylabel('theta-dot');
title('Value Iteration');
colorbar;
pause(0.5);

%% Build GP based estimate of value function using the computed policy and lookahead kernel
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
[grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
p_l = @(s) policy(p, grid_x, grid_x_dot, s);
% nu = min(dx,dx_dot);
nu = exp(-1);
sigma0 = 0.02;
sigmak = [dx; dx_dot; 1];
env2 = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
env_sim = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
gptd_l = GPTD_lookahead(env2, env_sim, nu, sigma0, sigmak, gamma_);
states = [reshape(grid_x,1,numPointsx*numPointsx_dot);...
          reshape(grid_x_dot,1,numPointsx*numPointsx_dot)];
max_episode_length = 100;
number_of_episodes = 1500;
debug_ = true;
gptd_l.build_posterior_monte_carlo(p_l, number_of_episodes, max_episode_length, debug_);
% V_l = gptd_l.get_value_function_frechet(states, p_l);
V_l = gptd_l.get_value_function(states, p_l);
figure;
imagesc(x,y,reshape(V_l,size(grid_x,1),size(grid_x_dot,2))');
xlabel('theta'); ylabel('theta-dot');
title('GPTD Lookahead Value function');
colorbar;
% hold on;
% scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
% hold off;

%% Build GP based estimate of value function using the computed policy
% dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
% dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
% [grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
% p_ = @(s) policy(p, grid_x, grid_x_dot, s);
% % nu = min(dx,dx_dot);
% nu = exp(-1) - 0.15;
% sigma0 = 0.02;
% sigmak = min(dx,dx_dot);
% env = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
% gptd = GPTD_fast(env, nu, sigma0, sigmak, gamma_);
% states = [reshape(grid_x,1,numPointsx*numPointsx_dot);...
%           reshape(grid_x_dot,1,numPointsx*numPointsx_dot)];
% max_episode_length = 100;
% number_of_episodes = 1500;
% debug_ = true;
% gptd.build_posterior_monte_carlo(p_, number_of_episodes, max_episode_length, debug_);
% gptd.visualize(grid_x,grid_x_dot);

% hold on;
% scatter(gptd.D(1,:),gptd.D(2,:),'MarkerFaceColor',[1 0 0],'LineWidth',1.5);
% hold off;

function a = policy(p, grid_x, grid_x_dot, s)
  a = interpn(grid_x, grid_x_dot, p, s(1), s(2));
end

            