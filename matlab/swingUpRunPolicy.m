function swingUpRunPolicy(continuous_time_dynamics, start, goal, tspan, name)
    
    opts = odeset('RelTol',1e-8,'AbsTol',1e-8);
    if (size(start,1)~=1 && size(start,2)~=1)
        return
    end
    if size(start,2)>1
        start = start';
    end
    [t,y] = ode45(@(t,y) continuous_time_dynamics(t,y), tspan, start, opts);
    figure;
    plot(y(:,1),y(:,2)); xlabel('theta'); ylabel('theta-dot');
    hold on;
    scatter(start(1),start(2),20,[0,1,0],'filled');
    text(start(1),start(2),'Start','horizontal','left','vertical','bottom');
    scatter(goal(1),goal(2),20,[1,0,0],'filled');
    text(goal(1),goal(2),'Goal','horizontal','left','vertical','bottom');
    title(name);
    hold off;

end