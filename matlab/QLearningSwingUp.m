clear;
close all;
clc;

%% Parameters
m = 1;
l = 1;
b = 0.5;
g = 1;
dt = 0.08;
numPointsx = 51;
numPointsx_dot = 51;
numPointsu = 21;
x_limits = [0,2*pi];   % Don't change limits for x as the values are wrapped around when an 
                       % action takes the system beyond 2pi or below 0 and any other
                       % limts physically don't make sense under this implementation
                       % If you change numPointsx ensure that goal x is a
                       % point on the grid i.e. x_limits(1):dx:x_limits(2) has goal x

x_dot_limits = [-2.2,2.2]; % Choose limits and numpoints for x_dot such that goal x_dot is a point on the grid 
                       % i.e. x_dot_limits(1):dx_dot:x_dot_limits(2) has goal_x_dot
Q = 10*eye(2);
u_limits = [-1,1];
R = 0.01;
gtol = 0.001;
goal = [pi;0]; 
start = [0;0];
gamma_ = 0.95;
epsilon = 0.1;
alpha = 0.1;
visualize = false;
test_policy = true;

%% Compute policy using value iteration
[p, V] = ValueIterationSwingUp(m, l, b, g, numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal', start', gamma_, gtol, dt, 30, 30, visualize, test_policy);

valueFig = figure();
subplot(2,1,1);
x = [x_limits(1), x_limits(2)];
y = [x_dot_limits(1), x_dot_limits(2)];
imagesc(x,y,V');
xlabel('theta'); ylabel('theta-dot');
title('Value Iteration');
colorbar;
pause(0.5);

%% Test VI policy on discrete environment
p = reshape(p, numPointsx*numPointsx_dot, 1);
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
du = (u_limits(2) - u_limits(1))/(numPointsu - 1);
[grid_x,grid_x_dot] = ndgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
state_grid = [reshape(grid_x, numPointsx*numPointsx_dot, 1), reshape(grid_x_dot, numPointsx*numPointsx_dot, 1)];
action_grid = u_limits(1):du:u_limits(2);

env_disc = pendulum_discrete(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, u_limits, numPointsu, Q, R, goal);
num_terms = 0;
for i=1:1:size(state_grid,1)
    env_disc.set(state_grid(i,:)');
    [s1,~,~] = env_disc.step(length(action_grid));
    env_disc.set(state_grid(i,:)');
    [s2,~,~] = env_disc.step(1);
    if ( (s1==i) && (s2==i) )
        disp('Action insufficient!');
    end
end

%% Compute policy using Q-Learning (discrete)

env = pendulum_discrete(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, u_limits, numPointsu, Q, R, goal);
env_test = pendulum_discrete(m, l, b, g, dt, x_limits, numPointsx, x_dot_limits, numPointsx_dot, u_limits, numPointsu, Q, R, goal);
max_episode_length = 500;
number_of_episodes = 15000;
test_every = number_of_episodes/200;
debug_ = true;
is_reward = false;

qlearn = QLearningDiscrete(env, env_test, gamma_, epsilon, alpha, is_reward);
% qlearn.train(number_of_episodes, max_episode_length, test_every, debug_);
qlearn.train_fixedstarts(state_grid', number_of_episodes, max_episode_length, test_every, debug_);

V_qlearn = qlearn.get_value_function();
subplot(2,1,2);
imagesc(reshape(V_qlearn, numPointsx_dot, numPointsx)');
colorbar;
xlabel('theta'); ylabel('theta-dot');
title('Q-Learning');

%%

qlearn_p = qlearn.get_policy();
for e = 1:1:10
    s = qlearn.test_env.reset();
    is_terminal = false;
    while (~is_terminal)
        [s,~,is_terminal] = qlearn.test_env.step(qlearn_p(s)); 
    end    
end