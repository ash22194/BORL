    close all;
    clear;
    clc;
    
%% Use Q-Learning to learn a policy
    is_slippery = false;
    train = true;
    map_size = [8,8];
%     map4x4_nohole = ['SSSS';
%                      'SSSS';
%                      'SSSS';
%                      'SSSG'];
%     map4x4 = map4x4_nohole;
    map4x4 = ['SSSS';
              'SHSH';
              'SSSH';
              'HSSG'];
    map8x8_nohole = ['SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSS';
                     'SSSSSSSG'];
    map8x8 = map8x8_nohole;
%     map8x8 = ['SSSSSSSS';
%               'SSSSSSSS';
%               'SSSHSSSS';
%               'SSSSSHSS';
%               'SSSHSSSS';
%               'SHHSSSHS';
%               'SHSSHSHS';
%               'SSSHSSSG'];
    map16x16 = ['SSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSS';
                'SSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSG'];
    map32x32 = ['SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSSSSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHSSHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHSSHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSSSSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHSSHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHSSHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSSSSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHSSHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHSSHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SSSSSHSSSSSSSHSSSSSSSHSSSSSSSHSS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSS';
                'SHHSSSHSSHHSSSHSSHHSSSHSSHHSSSHS';
                'SHSSHSHSSHSSHSHSSHSSHSHSSHSSHSHS';
                'SSSHSSSSSSSHSSSSSSSHSSSSSSSHSSSG'];
            
    if (all(map_size==[4,4]))
        env = frozenLake(map4x4, 4, is_slippery, false);    
    elseif (all(map_size==[8,8]))
        env = frozenLake(map8x8, 4, is_slippery, false);
    elseif (all(map_size==[16,16]))
        env = frozenLake(map16x16, 4, is_slippery, false);
    elseif (all(map_size==[32,32]))
        env = frozenLake(map32x32, 4, is_slippery, false);
    end
    filename = strcat('policy',int2str(map_size(1)),'x',int2str(map_size(2)),'.mat');
    gamma_ = 0.95;
    if (exist(filename,'file')==2 && ~train)
        policy = load(filename,'policy');
        policy = policy.policy;
    else
        alpha = 0.01;
        number_of_episodes = 4000;
        debug = true;
        [V,policy] = ValueIteration(env, 1:1:env.num_states, 1:1:env.num_actions, alpha, gamma_, number_of_episodes);  

        % Plot the value function
        imagesc(reshape(V,map_size));
        xlabel('x'); ylabel('y');
        title('Value Function - Value Iteration');
        colorbar;
        pause(0.5);

    %     figure;
    %     imagesc(reshape(policy,map_size));
    %     title('Policy')figure
    end
    
    
%% Use the learnt policy to build a posterior over the value function
    nu = exp(-1)-0.2;
    sigma0 = 0.1;
    sigmak = 0.5;
    if (all(map_size==[4,4]))
        env1 = frozenLake(map4x4, 4, is_slippery, true);    
    elseif (all(map_size==[8,8]))
        env1 = frozenLake(map8x8, 4, is_slippery, true);
    elseif (all(map_size==[16,16]))
        env1 = frozenLake(map16x16, 4, is_slippery, true);
    elseif (all(map_size==[32,32]))
        env1 = frozenLake(map32x32, 4, is_slippery, true);
    end
    gptd = GPTD_fast(env1, nu, sigma0, sigmak, gamma_);
    max_episode_length = 70;
    number_of_episodes = 1000;
    debug_ = true;
    policy_ = @(x)grid_policy(policy, map_size,x);
%     profile on;
    gptd.build_posterior_monte_carlo(policy_, number_of_episodes, max_episode_length, debug_);
    
%     profile viewer;save(strcat('GridWorldWorkspace',int2str(map_size(1)),'.mat'));
    [grid_x, grid_y] = meshgrid(1:map_size(1),1:map_size(2));
    states = [reshape(grid_x,1,map_size(1)*map_size(2));...
              reshape(grid_y,1,map_size(1)*map_size(2))];
%     gptd.build_posterior_vi(states, policy, number_of_episodes, debug_);
    V_gptd = gptd.get_value_function(states);
    
    figure
    imagesc(reshape(V_gptd,map_size));
    xlabel('x'); ylabel('y');
    title('Value Function - GPTD');
    colorbar;
    pause(0.5);
    
%     save(strcat('GridWorldWorkspace',int2str(map_size(1)),'.mat'));
    
%% Run GPSARSA to build policy

    sigma_s = [1;1];
    sigma_a = [1];
    nu = exp(-3/2)-0.15;
    epsilon = 0.1;
    epsilon_end = 0.05;
    actions = [1 2 3 4];
    is_reward = true;
    if (all(map_size==[4,4]))
        env2 = frozenLake(map4x4, 4, is_slippery, true);    
    elseif (all(map_size==[8,8]))
        env2 = frozenLake(map8x8, 4, is_slippery, true);
    elseif (all(map_size==[16,16]))
        env2 = frozenLake(map16x16, 4, is_slippery, true);
    elseif (all(map_size==[32,32]))
        env2 = frozenLake(map32x32, 4, is_slippery, true);
    end
    gpsarsa = GPSARSA_fast(env2, actions, true, epsilon, nu, sigma_s, sigma_a, sigma0, gamma_, is_reward);
    max_episode_length = 100;
    number_of_episodes = 3000;
    debug_ = true;
    profile on;
    gpsarsa.build_policy_monte_carlo_fixed_starts(states, number_of_episodes, max_episode_length, epsilon_end, debug_);
    profile viewer;
    V_gpsarsa = gpsarsa.get_value_function(states);
    figure
    imagesc(reshape(V_gpsarsa, map_size));
    xlabel('x'); ylabel('y');
    title('Value Function - GPSARSA');
    colorbar;
    pause(0.5);

%% Functions
    function p = grid_policy(policy, map_size, x)
            assert(x(1)<=map_size(1) && x(2)<=map_size(2), 'State out of bounds');
            assert(x(1)>0 && x(2)>0, 'State out of bounds');
            p = policy((x(1)-1)*map_size(2) + x(2));
    end
