close all;
clear;
clc;

%% Parameters
m = 1; mass_factor = 1.6;
l = 1; length_factor = 1.1;
b = 0.15;
g = 9.81;
dt = 0.001;
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
Q = [80,0;0,0.05];
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
imagesc(x,y,(V_bootstrapped - V)'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('Initial Difference')

%% Fit GP over the difference  - Using [theta, theta-dot] as "x" and difference in value function as "y"

% Generate training data
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
[grid_x, grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2), x_dot_limits(1):dx_dot:x_dot_limits(2));

num_train_points = 500;
x_train = [x_limits(1) + rand(num_train_points,1)*(x_limits(2) - x_limits(1)),...
           x_dot_limits(1) + rand(num_train_points,1)*(x_dot_limits(2) - x_dot_limits(1))];
       
y_train = interpn(grid_x, grid_x_dot, (V_bootstrapped - V), x_train(:,1), x_train(:,2));

% Test data - Grid?
x_query = [reshape(grid_x, numPointsx*numPointsx_dot, 1), reshape(grid_x_dot, numPointsx*numPointsx_dot, 1)];
meanfunc = [];
covfunc = @covSEard;
likfunc = @likGauss; 
hyp = struct('mean', [], 'cov', [0 0 0], 'lik', -1);
hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train);
[mu sigm] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x_train, y_train, x_query);

set(0,'CurrentFigure',valueFig);
subplot(4,1,4);
imagesc(x,y,reshape(mu, numPointsx, numPointsx_dot)'); colorbar;
xlabel('theta')
ylabel('theta-dot')
title('GPFit Difference')


