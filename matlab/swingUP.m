clear;
close all;
clc;

%% Parameters
dt = 0.1;
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
Q = eye(2);
u_limits = [-5,5];
R = 1;
gtol = 0.1;
goal = [pi,0]; 
start = [0,0];
gamma_ = 1;
maxiter = 1000;
max_policy_iter = 1000;
visualize = false;
test_policy = false;

%% Compute policy using value iteration
[policy, V] = ValueIterationSwingUp(numPointsx, numPointsx_dot, numPointsu, x_limits, x_dot_limits, u_limits,...
                               Q, R, goal, start, gamma_, gtol, dt, maxiter, max_policy_iter, visualize, test_policy);


imagesc(V);
xlabel('theta'); ylabel('theta-dot');
title('Policy Iteration');
colorbar;

%% Build GP based estimate of value function using the computed policy
dx = (x_limits(2)-x_limits(1))/(numPointsx - 1); 
dx_dot = (x_dot_limits(2)-x_dot_limits(1))/(numPointsx_dot - 1);
% du = (u_limits(2) - u_limits(1))/(numPointsu - 1);

nu = max(dx,dx_dot);
sigma0 = 0;
sigmak = nu;
gptd = GPTD(nu, sigma0, sigmak, gamma_);
max_epsiode_length = 50;
number_of_episodes = 150;

% Grid of states
[grid_x,grid_x_dot] = meshgrid(x_limits(1):dx:x_limits(2),x_dot_limits(1):dx_dot:x_dot_limits(2));
goal_grid = [0,0];
goal_grid(2) = find(grid_x(1,:)==goal(1));
goal_grid(1) = find(grid_x_dot(:,1)==goal(2));

% Vector of possible actions
% possibleActions = u_limits(1):du:u_limits(2);

% Calculate position cost
positionCost = zeros(size(grid_x_dot,1),size(grid_x,2));
for i = 1:1:length(grid_x(1,:))
    for j = 1:1:length(grid_x_dot(:,1))
        relP = [grid_x(j,i),grid_x_dot(j,i)] - goal;
        positionCost(j,i) = (relP)*Q*relP';
    end
end

for i=1:1:number_of_episodes
  disp(strcat('Episode number : ',int2str(i)));
  % Pick a random start
  s = [x_limits(1)+rand()*(x_limits(2) - x_limits(1)); x_dot_limits(1)+rand()*(x_dot_limits(2) - x_dot_limits(1))];
  num_steps = 0;
  disp(strcat('Dictionary size : ',int2str(size(gptd.D,2))));
  
  while (num_steps < max_epsiode_length)
      
      u = interp2(grid_x, grid_x_dot, policy, s(1), s(2));
      s_ = zeros(2,1);
      s_(1) = s(1) + dt*s(2);
      s_(2) = s(2) - dt*(s(2) + sin(s(1))) + dt*u;
      
      % Check for wrap around in x
      greaterThan = s_(1) > x_limits(2);
      s_(1) = s_(1) - x_limits(2)*greaterThan;
      lessThan = s_(1) < x_limits(1);
      s_(1) = s_(1) + x_limits(2)*lessThan;
      
      % Check for bounds in x_dot
      greaterThan = s_(2) > x_dot_limits(2);
      s_(2) = s_(2) + (x_dot_limits(2) - s_(2))*greaterThan;
      lessThan = s_(2) < x_dot_limits(1);
      s_(2) = s_(2) + (x_dot_limits(1) - s_(2))*lessThan;
      
      % Calculate the reward/cost
      r = (interp2(grid_x, grid_x_dot, positionCost, s_(1), s_(2)) + R*u^2)*dt;
      
      if (((s_(1) - goal(1)) < dx) && ((s_(2) - goal(2)) < dx_dot))
        disp('Reached goal!');
        gptd = gptd.update(s_,s,0);
        break;
      end
      gptd = gptd.update(s_,s,r);
      s = s_;
      num_steps = num_steps + 1;
  end
end
 
gptd.visualize(grid_x,grid_x_dot);