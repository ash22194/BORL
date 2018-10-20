    close all;
    clear;
    clc;
    
%% Use Q-Learning to learn a policy
    is_slippery = false;
    train = false;
    map_size = [16,16];
    map4x4 = ['SSSS';
              'SHSH';
              'SSSH';
              'HSSG'];
    map8x8 = ['SSSSSSSS';
              'SSSSSSSS';
              'SSSHSSSS';
              'SSSSSHSS';
              'SSSHSSSS';
              'SHHSSSHS';
              'SHSSHSHS';
              'SSSHSSSG'];
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
        env = frozenLake(map4x4, 4, is_slippery);    
    elseif (all(map_size==[8,8]))
        env = frozenLake(map8x8, 4, is_slippery);
    elseif (all(map_size==[16,16]))
        env = frozenLake(map16x16, 4, is_slippery);
    elseif (all(map_size==[32,32]))
        env = frozenLake(map32x32, 4, is_slippery);
    end
    filename = strcat('policy',int2str(map_size(1)),'x',int2str(map_size(2)),'.mat');
    gamma_ = 0.95;
    if (exist(filename,'file')==2 && ~train)
        policy = load(filename,'policy');
        policy = policy.policy;
    else
        alpha = 0.01;
        epsilon = 0.1;
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
    nu = 0;
    sigma0 = 0;
    sigmak = sqrt(0.2);
    gptd = GPTD(env, nu, sigma0, sigmak, gamma_);
    max_episode_length = 70;
    number_of_episodes = 150;
    debug_ = true;
%     profile on;
%     gptd.build_posterior_monte_carlo(policy, number_of_episodes, max_episode_length, debug_);
    gptd.build_posterior_vi(1:1:(map_size(1)*map_size(2)), policy, number_of_episodes, debug_);
%     profile viewer;save(strcat('GridWorldWorkspace',int2str(map_size(1)),'.mat'));
    V_gptd = gptd.get_value_function(1:1:env.num_states);
    
    figure
    imagesc(reshape(V_gptd,map_size));
    xlabel('x'); ylabel('y');
    title('Value Function - GPTD');
    colorbar;
    pause(0.5);
    
    save(strcat('GridWorldWorkspace',int2str(map_size(1)),'.mat'));