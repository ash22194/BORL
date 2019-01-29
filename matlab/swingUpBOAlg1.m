clear;
close all;
clc;

%% Parameters
m = 1; mass_factor = 2;
l = 1; length_factor = 1;
b = 0.5;
g = 9.81;
dt = 0.1;
numPointsx = 51;
numPointsx_dot = 81;
numPointsu = 61;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-8,8]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = eye(2);
u_limits = [-10,10];
R = 0.1;
gtol = 0.0001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 1;
maxiter = 500;
max_policy_iter = 30;
visualize = false;
test_policy = false;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(mass_factor*m, length_factor*l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);
valueFig = figure();
subplot(3,1,1);
imagesc(V);
xlabel('theta-dot'); ylabel('theta');
title('Value Iteration - Target');
colorbar;
pause(0.5);

%% Build Q value estimates in "simulation"
[p_init, V_init] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);

set(0,'CurrentFigure',valueFig);
subplot(3,1,2);
imagesc(V_init);
xlabel('theta-dot'); ylabel('theta');
title('Value Iteration - Initialization');
colorbar;
pause(0.5);

dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
du  = (u_limits(2)-u_limits(1))/(numPointsu-1);
[grid_x, grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
states = [reshape(grid_x,size(grid_x,1)*size(grid_x,2),1), reshape(grid_x_dot,size(grid_x_dot,1)*size(grid_x_dot,2),1)];
actions = [u_limits(1):du:u_limits(2)]';

%-- Test learned policy on target environment --%
% continuous_time_dynamics = @(t,y) gridBasedSwingUp(t,y,mass_factor*m,length_factor*l,b,g,grid_x,grid_x_dot,p_init,x_limits,x_dot_limits);
% swingUpRunPolicy(continuous_time_dynamics, start, goal, [0,30], strcat('Vanilla Policy, M Factor:',num2str(mass_factor),', L Factor:',num2str(length_factor)))
%-- --%

env = pendulum(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);

dynamics = @(s,a)env.dynamics(s,a);
reward = @(s_,a)env.cost(s_,a);

V_interpolant = @(s) interpn(grid_x, grid_x_dot, reshape(V_init, numPointsx, numPointsx_dot), s(:,1), s(:,2));
Q_init = buildQFunction(states, actions, dynamics, reward, V_interpolant, gamma_);

%% Run BOAlg1 for a slightly different system with the bootstrapped muQ

state_grid(:,:,1) = grid_x;
state_grid(:,:,2) = grid_x_dot;
sigma_l = [du];
sigma_n = 10^(-2);
sigma_f = 1;
QKernel = SqExpKernel(actions, sigma_l, sigma_n, sigma_f);

muQ_bootstrapped = @(s,a) muQ(reshape(Q_init, numPointsx, numPointsx_dot, numPointsu), state_grid, actions, s, a, QKernel);
sigmaQ_bootstrapped = @(s,a) sigmaQ(actions, s, a, QKernel);

env1 = pendulum(mass_factor*m, length_factor*l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
sigma_s = [0.2*dt; 0.2];
sigma_a = [0.1];
num_episodes = 50;
horizon = 50;
debug_ = true;

borl = BORLAlg1(env1, u_limits, gamma_, sigma_s, sigma_a, muQ_bootstrapped, sigmaQ_bootstrapped);
borl.train(num_episodes, horizon, debug_)
Q_borl = zeros(size(states,1),1);
Q_variance = zeros(size(states,1),1);
num_actions_to_sample = 10;
actions_sampled = u_limits(1) + (u_limits(2) - u_limits(1))*rand(num_actions_to_sample,1);
policy_borl = zeros(size(states),size(actions,2));
for s=1:1:size(states,1)
    Qa = zeros(size(actions_sampled,1),1);
    for a=1:1:size(actions_sampled,1)
        Qa(a,1) = borl.getQ(states(s,:)',actions_sampled(a,:)', 1);
    end
    [Q_borl(s,1), minA] = min(Qa);
    Q_variance(s,1) = borl.getQvariance(states(s,:)', actions_sampled(minA,:), 1);
    policy_borl(s,:) = actions_sampled(minA,:);
    s
end
%-- Test final policy --%
continuous_time_dynamics = @(t,y) gridBasedSwingUp(t,y,mass_factor*m,length_factor*l,b,g,grid_x,grid_x_dot,...
    reshape(policy_borl,numPointsx,numPointsx_dot),x_limits,x_dot_limits);
swingUpRunPolicy(continuous_time_dynamics, start, goal, [0,10], strcat('Final Policy, M Factor:',num2str(mass_factor),', L Factor:',num2str(length_factor)));
%-- --%

set(0,'CurrentFigure',valueFig);
subplot(3,1,3);
imagesc(reshape(Q_borl,numPointsx,numPointsx_dot));
title('BORLAlg1 - Final');
xlabel('theta-dot'); ylabel('theta');
colorbar;

function muQ_ = muQ(Q_init, states, actions, state, action, QKernel)
    % state - Sx1
    % action - Ax1
    
    muQ_vec = zeros(size(actions,1),1);
    for a = 1:1:size(actions,1)
        muQ_vec(a,1) = interpn(states(:,:,1), states(:,:,2), Q_init(:,:,a), state(1), state(2));
    end
    ka = QKernel.distance(actions, repmat(action',size(actions,1),1));
    muQ_ = ka'*QKernel.K_inv*muQ_vec; 
end

function sigmaQ_ = sigmaQ(actions, state, action, QKernel)
    % state - Sx1
    % action - Ax1
    
    ka = QKernel.distance(actions, repmat(action',size(actions,1),1));
    kaa = QKernel.distance(action', action');
    sigmaQ_ = kaa - ka'*QKernel.K_inv*ka;
end
