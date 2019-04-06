close all;
clear;
clc;

%% Parameters
m = 1; mass_factor = 1.6;
l = 1; length_factor = 1.1;
b = 0.15;
g = 9.81;
dt = 0.005;
numPointsx = 51;
numPointsx_dot = 81;
numPointsu = 121;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(1) has goal x

x_dot_limits = [-6.5,6.5]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = [40,0;0,0.02];
u_limits = [-25,25];
R = 0.02;
gtol = 0.001;
goal = [pi;0];
start = [0;0];
gamma_ = 0.99;
maxiter = 600;
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

set(0,'CurrentFigure',valueFig);
subplot(4,1,3);
imagesc(x,y,(V - V_bootstrapped)'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('Initial Difference')

%% Fit GP over Value function  - Using [theta, theta-dot] as "x" and difference in value function as "y"

% target = V;
% 
% % Generate training data
% dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
% dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
% [grid_x, grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2), x_dot_limits(1):dx_dot:x_dot_limits(2));
% 
% num_train_points = 2000;
% x_train = [x_limits(1) + rand(num_train_points,1)*(x_limits(2) - x_limits(1)),...
%            x_dot_limits(1) + rand(num_train_points,1)*(x_dot_limits(2) - x_dot_limits(1))];
%  
% y_train = interpn(grid_x, grid_x_dot, target, x_train(:,1), x_train(:,2));
% % x_train = [x_train, interpn(grid_x, grid_x_dot, (V_bootstrapped), x_train(:,1), x_train(:,2))];
% % x_train = interpn(grid_x, grid_x_dot, (V_bootstrapped), x_train(:,1), x_train(:,2));
% 
% % Test data - Grid?
% x_query = [reshape(grid_x, numPointsx*numPointsx_dot, 1), ...
%            reshape(grid_x_dot, numPointsx*numPointsx_dot, 1)];
% % x_query = [reshape(grid_x, numPointsx*numPointsx_dot, 1), ...
% %            reshape(grid_x_dot, numPointsx*numPointsx_dot, 1),...
% %            reshape(V_bootstrapped, numPointsx*numPointsx_dot, 1)];
% % x_query = reshape(V_bootstrapped, numPointsx*numPointsx_dot, 1);
% meanfunc = {'meanZero'};
% % grid(:,:,1) = grid_x;
% % grid(:,:,2) = grid_x_dot;
% % meanfunc = {@meanInterp2, grid, V_bootstrapped};
% % covfunc = {@covSE, 'ard', []};
% covfunc = @covSEard;
% likfunc = @likGauss; 
% hyp = struct('mean', [], 'cov', zeros(1, size(x_query,2)+1), 'lik', -1);
% regularizer = struct('function', @regularizer_squared, 'lambda', 20);
% hyp = minimize(hyp, @gp, -100, regularizer, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train);
% [mu, sigm] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train, x_query);
% 
% set(0,'CurrentFigure',valueFig);
% subplot(4,1,4);
% imagesc(x,y,(reshape(mu, numPointsx, numPointsx_dot))'); colorbar;
% xlabel('theta')
% ylabel('theta-dot')
% title('GPFit')
% 
% fiterror = sqrt(mean((mu - reshape(target, numPointsx*numPointsx_dot, 1)).^2));
% disp(strcat('Mean Squared Error: ',num2str(fiterror)));

%% %% Fit GP over Q function  - Using [theta, theta-dot] as "x" and difference in value function as "y"

% Generate training data
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
du  = (u_limits(2)-u_limits(1))/(numPointsu-1);

env = pendulum(m*mass_factor, l*length_factor, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, numPointsu, Q, R, goal);
dynamics = @(s,a)env.dynamics(s,a);
reward = @(s_,a)env.cost(s_,a);

[grid_x, grid_x_dot, grid_actions] = ndgrid(x_limits(1):dx:x_limits(2), x_dot_limits(1):dx_dot:x_dot_limits(2), u_limits(1):du:u_limits(2));

states = [reshape(grid_x(:,:,1),size(grid_x,1)*size(grid_x,2),1), reshape(grid_x_dot(:,:,1),size(grid_x_dot,1)*size(grid_x_dot,2),1)];
actions = [u_limits(1):du:u_limits(2)]';

V_interpolant_init = @(s) interpn(grid_x(:,:,1), grid_x_dot(:,:,1), reshape(V_bootstrapped, numPointsx, numPointsx_dot), s(:,1), s(:,2));
Q_init = buildQFunction(states, actions, dynamics, reward, V_interpolant_init, gamma_);
Q_init = reshape(Q_init, numPointsx, numPointsx_dot, numPointsu);

V_interpolant_final = @(s) interpn(grid_x(:,:,1), grid_x_dot(:,:,1), reshape(V, numPointsx, numPointsx_dot), s(:,1), s(:,2));
Q_final = buildQFunction(states, actions, dynamics, reward, V_interpolant_final, gamma_);
Q_final = reshape(Q_final, numPointsx, numPointsx_dot, numPointsu);

target = Q_final;

num_train_points = 2000;
x_train = [x_limits(1) + rand(num_train_points,1)*(x_limits(2) - x_limits(1)),...
           x_dot_limits(1) + rand(num_train_points,1)*(x_dot_limits(2) - x_dot_limits(1)),...
           u_limits(1) + rand(num_train_points,1)*(u_limits(2) - u_limits(1))];

% x_index_sample = randi(size(grid_x, 1), num_train_points, 1);
% x_dot_index_sample = randi(size(grid_x_dot, 2), num_train_points, 1);
% x_train = [reshape(grid_x(x_index_sample, 1, 1), num_train_points, 1), ...
%            reshape(grid_x_dot(1, x_dot_index_sample, 1), num_train_points, 1), ...
%            p_bootstrapped(x_index_sample + (x_dot_index_sample-1)*size(grid_x,1))];

y_train = interpn(grid_x, grid_x_dot, grid_actions, target, x_train(:,1), x_train(:,2), x_train(:,3));
% x_train = [x_train, interpn(grid_x, grid_x_dot, grid_actions, Q_init, x_train(:,1), x_train(:,2), x_train(:,3))];

% y_train = interpn(grid_x(:,:,1), grid_x_dot(:,:,1), target, x_train(:,1), x_train(:,2));
% x_train = [x_train, interpn(grid_x, grid_x_dot, (V_bootstrapped), x_train(:,1), x_train(:,2))];
% x_train = interpn(grid_x, grid_x_dot, (V_bootstrapped), x_train(:,1), x_train(:,2));

% Test data - Grid?
x_query = [reshape(grid_x(:,:,1), numPointsx*numPointsx_dot, 1), ...
           reshape(grid_x_dot(:,:,1), numPointsx*numPointsx_dot, 1), ...
           reshape(p, numPointsx*numPointsx_dot, 1)];
% x_query = [x_query(:,1:2), reshape(p, numPointsx*numPointsx_dot,1), interpn(grid_x, grid_x_dot, grid_actions, Q_init, x_query(:,1), x_query(:,2), x_query(:,3))];
target = V;       
meanfunc = {'meanZero'};
% grid(:,:,1) = grid_x;
% grid(:,:,2) = grid_x_dot;
% meanfunc = {@meanInterp2, grid, V_bootstrapped};
% covfunc = {@covSE, 'ard', []};
covfunc = @covSEard;
likfunc = @likGauss; 
hyp = struct('mean', [], 'cov', zeros(1, size(x_query,2)+1), 'lik', -1);
regularizer = struct('function', @regularizer_squared, 'lambda', 20);
hyp = minimize(hyp, @gp, 500, regularizer, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train);
[mu, sigm] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train, x_query);

set(0,'CurrentFigure',valueFig);
subplot(4,1,4);
imagesc(x,y,(reshape(mu, numPointsx, numPointsx_dot))'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('GPFit')

fiterror = sqrt(mean((mu - reshape(target, numPointsx*numPointsx_dot, 1)).^2));
disp(strcat('Mean Squared Error: ',num2str(fiterror)));
