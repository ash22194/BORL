clear;
close all;
clc;

%% Parameters
m = 1; mass_factor = 1.6;
l = 1; length_factor = 1.1;
b = 0.15;
g = 9.81;
dt = 0.005;
numPointsx = 51;
numPointsx_dot = 81;
numPointsu = 73;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-6.5,6.5]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = [40,0;0,0.02];
u_limits = [-15,15];
R = 0.02;
gtol = 0.001;
goal = [pi;0];
start = [0;0];
gamma_ = 0.99;
maxiter = 600;
max_policy_iter = 30;
visualize = false;
test_policy = true;

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

set(0,'CurrentFigure',valueFig);
subplot(4,1,3);
imagesc(x,y,abs(V_bootstrapped - V)'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('Initial Difference');
disp(strcat('Initial absolute mean error', num2str(mean(mean(abs(V_bootstrapped - V))))));

%% Build GP based estimate of value function using the computed policy
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
du  = (u_limits(2)-u_limits(1))/(numPointsu-1);

[grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
p_target = @(s) policy(p, grid_x, grid_x_dot, s);
p_init = @(s) policy(p_bootstrapped, grid_x, grid_x_dot, s);

env = pendulum(m*mass_factor, l*length_factor, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
env_sim = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
discra = false;
is_reward = false;
dynamics = @(s,a)env.dynamics(s,a);
reward = @(s_,a)env.cost(s_,a);
states = [reshape(grid_x,numPointsx*numPointsx_dot,1), reshape(grid_x_dot,numPointsx*numPointsx_dot,1)];
actions = [u_limits(1):du:u_limits(2)]';

V_interpolant = @(s) interpn(grid_x, grid_x_dot, V_bootstrapped, s(:,1), s(:,2));
Q_init = buildQFunction(states, actions, dynamics, reward, V_interpolant, gamma_);
% Q_bootstrapped = @(s,a) Q_interp(reshape(Q_init, numPointsx, numPointsx_dot, numPointsu), ...
%                                  repmat(grid_x,1,1,numPointsu), repmat(grid_x_dot,1,1,numPointsu), ...
%                                  repmat(reshape(actions,1,1,size(actions,1)), numPointsx, numPointsx_dot, 1), s, a);
Q_bootstrapped = @(s,a) Q_interp(reshape(Q_init, numPointsx, numPointsx_dot, numPointsu), ...
                                 x_limits(1):dx:x_limits(2), x_dot_limits(1):dx_dot:x_dot_limits(2), actions', s, a);

sigma0 = 0.02;
sigma_s = [0.5977; 1.9957];
sigma_a = 5.7314;
sigmaf = 13.6596;

sigma_q = 4.298;

% sigma_s = [1.3087; 2.9121];
% sigma_a = 9.6583;
% sigma_q = 7.0756;
% sigmaf = 16.8202;
nu = (exp(-1)-0.355)*sigmaf^2;

epsilon = 0.1;
epsilon_end = 0.05;
gpsarsa_b = GPSARSA_episodic(env, Q_bootstrapped, u_limits, discra, epsilon, nu, sigma_s, sigma_a, sigma_q, sigma0, sigmaf, gamma_, is_reward);
max_episode_length = 1000;
number_of_episodes = 2000;
debug_ = true;
profile on;
gpsarsa_b.build_policy_monte_carlo(number_of_episodes, max_episode_length, epsilon_end, debug_);
profile viewer;
[V_b,policy_b] = gpsarsa_b.get_value_function(states');

set(0,'CurrentFigure',valueFig);
subplot(4,1,4);
imagesc(x,y,abs(reshape(V_b, numPointsx, numPointsx_dot) - V)'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('Final Difference');
disp(strcat('Final absolute mean error', num2str(mean(mean(abs(reshape(V_b, numPointsx, numPointsx_dot) - V))))));

%% Vanilla GPTD

% gptd = GPTD_fast(env, nu, sigma0, sigmak, gamma_);
% gptd.build_posterior_monte_carlo_fixed_starts(p_target, states', number_of_episodes, max_episode_length, debug_);
% V_gptd = gptd.get_value_function(states');
% 
% set(0,'CurrentFigure',valueFig);
% subplot(4,1,4);
% imagesc(x, y, reshape(V_gptd,size(grid_x,1),size(grid_x_dot,2))');
% xlabel('theta'); ylabel('theta-dot');
% title('Vanilla GPTD');
% colorbar;

function a = policy(p, grid_x, grid_x_dot, s)
  a = interpn(grid_x, grid_x_dot, p, s(1), s(2));
end

function Q_bootstrapped = Q_interp(Q, grid_x, grid_x_dot, grid_actions, state, action)
%   Q_bootstrapped = interpn(grid_x, grid_x_dot, grid_actions, Q, state(1), state(2), action);
    Q_bootstrapped = lininterpn(grid_x, grid_x_dot, grid_actions, Q, state(1), state(2), action);
end
