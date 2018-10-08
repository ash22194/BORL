    close all;
    clear;
    clc;
    
%% Use Q-Learning to learn a policy
    is_slippery = false;
    map_size = [4,4];
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
    if (all(map_size==[4,4]))
        env = frozenLake(map4x4, 4, is_slippery);
    elseif (all(map_size==[8,8]))
        env = frozenLake(map8x8, 4, is_slippery);
    end
    gamma_ = 1.0;
    alpha = 0.01;
    epsilon = 0.1;
    num_episodes = 100000;
    max_length_episode = 300;
    test_every = 10;
    debug = false;
    agent = QLearningDiscrete(env, gamma_, epsilon, alpha);
    agent.train(num_episodes, max_length_episode, test_every, debug);
    policy = agent.get_policy(1:1:env.num_states);
    V = agent.get_value_function(1:1:env.num_states);
    
    % Plot the value function
    imagesc(reshape(V,map_size));
    xlabel('x'); ylabel('y');
    title('Value Function - QLearning');
    colorbar;
    pause(0.5);

%% Use the learnt policy to build a posterior over the value function
    nu = 0;
    sigma0 = 0;
    sigmak = 0.2;
    gptd = GPTD(env, nu, sigma0, sigmak, gamma_);
    max_episode_length = 50;
    number_of_episodes = 150;
    debug_ = true;
    gptd.build_posterior(policy, number_of_episodes, max_episode_length, debug_);
    V_gptd = gptd.get_value_function(1:1:env.num_states);
    
    figure
    imagesc(reshape(V_gptd,map_size));
    xlabel('x'); ylabel('y');
    title('Value Function - GPTD');
    colorbar;
    pause(0.5);