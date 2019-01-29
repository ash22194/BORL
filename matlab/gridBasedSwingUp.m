function dx = gridBasedSwingUp(t,x,m,l,b,g,grid_x,grid_x_dot,policy,x_limits,x_dot_limits)
x_ = x;
% Wrap around
if (x_(1) < x_limits(1))
    x_(1) = x_(1) + x_limits(2);
end
if (x_(1) > x_limits(2))
    x_(1) = x_(1) - x_limits(2);
end
% Threshold velocities
if (x_(2) < x_dot_limits(1))
    x_(2) = x_dot_limits(1);
end
if (x_(2) > x_dot_limits(2))
    x_(2) = x_dot_limits(2);
end
u = interpn(grid_x,grid_x_dot,policy,x_(1),x_(2));
dx = zeros(2,1);
dx(1) = x(2);
dx(2) = (u - m*g*l*sin(x(1)) - b*x(2))/(m*l^2);
end