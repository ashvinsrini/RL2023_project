

import yaml
import gymnasium as gym

import numpy as np 
from types import SimpleNamespace as SN
from pathlib import Path
import copy
import utils.common_utils as cu
from algos.ddpg_extension import DDPGExtension
from algos.ddpg_agent import DDPGAgent
#from algos.ppo_agent import PPOAgent
from utils.recorder import RecordVideo
from tqdm import tqdm

# Function to test a trained policy
def test(agent, env_name, algo_name):
    # Load model
    agent.load_model()
    print("Testing...")
    total_test_reward, total_test_len = 0, 0
    returns = []
    
    cur_dir=Path().cwd()
    cfg_path= cur_dir/'cfg'
    # read configuration parameters:
    cfg={'cfg_path': cfg_path, 'algo_name': algo_name,  "device":"cuda"}
    env_cfg=yaml.safe_load(open(cfg_path /'envs'/f'{env_name}_env.yaml', 'r'))
    
    # prepare folders to store results
    work_dir = cur_dir/'results'/env_cfg["env_name"]/algo_name
    video_test_dir=work_dir/"video"/"test"
    
    for ep in range(agent.cfg.test_episodes):
        frames = []
        seed = np.random.randint(low=1, high=1000)
        observation, _ = agent.env.reset(seed=seed)
        test_reward, test_len, done = 0, 0, False
        
        while not done and test_len < agent.cfg.max_episode_steps:
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, truncated, info = agent.env.step(action.flatten())
            fs = agent.env.render()
            frames = frames+fs
            test_reward += reward
            test_len += 1
        total_test_reward += test_reward
        total_test_len += test_len
        returns.append(test_reward)
        
        if ep%100==0:
            cu.save_rgb_arrays_to_gif(frames, video_test_dir/('_seed_'+str(agent.seed)+'_ep_'+str(ep)+'.gif'))

    print(f"Average test reward over {len(returns)} episodes: {total_test_reward/agent.cfg.test_episodes},+- {np.std(np.array(returns))}; \
        Average episode length: {total_test_len/agent.cfg.test_episodes}")
    return [f"Average test reward over {len(returns)} episodes: {total_test_reward/agent.cfg.test_episodes},+- {np.std(np.array(returns))}; \
        Average episode length: {total_test_len/agent.cfg.test_episodes}", {total_test_reward/agent.cfg.test_episodes}, {np.std(np.array(returns))}, {total_test_len/agent.cfg.test_episodes}]

# Setup: read the configurations and generate the environment.
def setup(algo=None, env='easy', cfg_args={}, render=True, train_episodes=None):
    # set the paths
    cur_dir=Path().cwd()
    cfg_path= cur_dir/'cfg'
    
    # read configuration parameters:
    cfg={'cfg_path': cfg_path, 'algo_name': algo,  "device":"cuda"}
    env_cfg=yaml.safe_load(open(cfg_path /'envs'/f'{env}_env.yaml', 'r'))
    algo_cfg=yaml.safe_load(open(cfg_path /'algo'/f'{algo}.yaml', 'r'))
    cfg.update(env_cfg)
    cfg.update(algo_cfg)
    cfg.update(cfg_args)
    
    # forcely change train_episodes
    if train_episodes is None:
        True
    else:
        cfg["train_episodes"] = train_episodes
    
    # prepare folders to store results
    work_dir = cur_dir/'results'/cfg["env_name"]/str(algo)
    model_dir=work_dir/"model"
    logging_dir=work_dir/"logging"
    video_train_dir=work_dir/"video"/"train"
    video_test_dir=work_dir/"video"/"test"
    for dir in [work_dir, model_dir, logging_dir, video_train_dir, video_test_dir]:
        cu.make_dir(dir)
        
    cfg.update({'work_dir':work_dir, "model_dir":model_dir, "logging_dir": logging_dir, "video_train_dir": video_train_dir, "video_test_dir": video_test_dir})
    cfg = SN(**cfg)
    
    # set seed
    if cfg.seed == None:
        seed = np.random.randint(low=1, high=1000)
    else:
        seed = cfg.seed
    
    ## Create environment
    env=cu.create_env(cfg_path /'envs'/f'{env}_env.yaml')

   
    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = cfg.video_test_dir
        # During training, save every 50th episode
        else:
            ep_trigger = 1000   # Save video every 50 episodes
            video_path = cfg.video_train_dir
        
        if render:
            env = RecordVideo(
                env, video_path,
                episode_trigger=lambda x: x % ep_trigger == 0,
                name_prefix=cfg.exp_name)


    eval_env=copy.deepcopy(env)
    env.reset(seed=seed) # we only set the seed here. During training, we don't have to set the seed when performing reset().
    eval_env.reset(seed=seed+1000)
    eval_env=None # For simplicity, we don't evaluate the performance during training.
        
    # Get dimensionalities of actions and observations
    action_space_dim = cu.get_space_dim(env.action_space)
    observation_space_dim = cu.get_space_dim(env.observation_space)
    
    config={
        "args": cfg,
        "env":env,
        "eval_env":eval_env,
        "action_space_dim": action_space_dim,
        "observation_space_dim": observation_space_dim,
        "seed":seed,
        "device": 'cuda'
    }
    return config


# Task 1.1: Train each agents' performance
# Implement your algorithm (either DDPG or PPO) in algo/ddpg.py or algo/ppo.py
# After the implementation, train your algorithm with the following code
# Train the algorithm in all three environments
# The code will train the algorithm with 3 random seeds
# Your code must be compatible with the following provided python code
# Below, you will find an example of how to test your code
train_episodes = 1000  # Limit the number of training episode for a fast test

config=setup(algo='ddpg', env='easy', train_episodes=train_episodes, render=False)

config["seed"] = 43
config["batch_size"] = 1024

result_med = {"seed": [], "log": [], "reward" : [], "std": [], "len": [], "theta": [], "sigma": []}
result_dif = {"seed": [], "log": [], "reward" : [], "std": [], "len": [], "theta": [], "sigma": []}

theta_ = np.linspace(0,1,20)
sigma_ = np.linspace(0,1,20)
for theta in theta_:
    for sigma in sigma_:

        implemented_algo = 'ddpg_extension'# choose 'ppo_extension' or 'ddpg_extension'
        environment = 'middle'

        training_seeds = [1,2,3]
        for i in range(3):
            config=setup(algo=implemented_algo, env=environment)
            config["theta"] = theta 
            config["sigma"] = sigma
            param = "_" + str(theta) + "_" + str(sigma)
            print ("<<<<<< PARAM >>>>>>>", param)
            config["seed"] = i
            training_seeds.append(i)


            if config["args"].algo_name == 'ppo_extension':
                agent=PPOExtension(config)
            elif config["args"].algo_name == 'ddpg_extension':
                agent=DDPGExtension(config)
            else:
                raise Exception('Please use ppo or ddpg!')

            # Train the agent using selected algorithm    
            agent.train()


        # **Test**: After training, run the following code to test your agents.

        training_seeds = []
        for i in range(3):
            config=setup(algo=implemented_algo, env=environment, render=False)
            config["theta"] = theta 
            config["sigma"] = sigma
            config["seed"] = i
            training_seeds.append(i)


            if config["args"].algo_name == 'ppo_extension':
                agent=PPOExtension(config)
            elif config["args"].algo_name == 'ddpg_extension':
                agent=DDPGExtension(config)
            else:
                raise Exception('Please use ppo or ddpg!')

            # Test the agent in the selected environment
            output = test(agent, environment, implemented_algo)
            result_med["log"].append(output[0])
            result_med["reward"].append(output[1])
            result_med["std"].append(output[2])
            result_med["len"].append(output[3])
            result_med["seed"].append(i)
            result_med["theta"].append(theta)
            result_med["sigma"].append(sigma)

        # Task 2.2: Plot improved algorithm performance¶


        ## Run the following code to plot PPO or DDPG's training performances
        # import warnings
        # warnings.filterwarnings('ignore')

        # Uncomment the algorithm you chose 
        implemented_algo =  'ddpg_extension'
        # 'ppo_extension' or 'ddpg_extension'
        environment = 'middle'

        # Loop over the three difficulty levels

        training_seeds = [0,1,2]

        config=setup(algo=implemented_algo, env=environment, render=False)
        config["theta"] = theta 
        config["sigma"] = sigma
        config["seed"] = 0

        agent= DDPGExtension(config) # or PPOExtension(config)

        # plot the statistical training curves with specific random seeds
        cu.plot_algorithm_training(agent.logging_dir, training_seeds, agent.env_name, implemented_algo)


        # ## Task 2.3: Comparison of Improved and Original Algorithm Performance

        ## Run the following code to draw the comparison plots of PPO and DDPG's training performances
        import warnings
        warnings.filterwarnings('ignore')

        environment = 'middle'

        orgin_alo_name = 'ddpg' #or 'ppo'
        improved_alo_name =  'ddpg_extension'# or 'ppo_extension'

        config=setup(algo=orgin_alo_name, env=environment, render=False)
        origin_agent = DDPGAgent(config)# or PPOAgent(config)

        config=setup(algo=improved_alo_name, env=environment, render=False)
        config["theta"] = theta 
        config["sigma"] = sigma
        improved_agent = DDPGExtension(config)# or PPOExtension(config)

        # make the comparison plot
        cu.compare_algorithm_training(origin_agent, improved_agent, seeds=[0,1,2], param = param)


        # <a id='Q1'></a>
        # <div class=" alert alert-warning">
        #     <h3><b>Student Question 1</b> (30 points) </h3> 
        #     Explain how you extended PPO/DDPG and why in a maximum of 200 words. In addition, explain briefly in which parts of the source code the changes are (refer to file name and function names or lines of code).
        # </div>

        # DOUBLE CLICK HERE TO EDIT, CLEAR THIS TEXT AND ANSWER HERE

        # <a id='T3'></a>
        # <div class=" alert alert-warning">
        #     <h3><b>Student Task 3</b> (+20 points) </h3>
        #     This task give bonus points to the project works that get highest performance in the difficult environment. At the end of the course, we will use everyone's improved agent (please submit your pretrained weights) to run the competition on the most difficult sanding environment. Competitive grading: all projects are evaluated in the difficult environment for performance and put into ranking order. Top 10% of submitted projects get bonus points. Best performing project (100% ranked) gets 20 bonus points, 95% ranked gets 10 bonus points, 90% or lower ranked get 0 bonus points.
        # </div>

        # ## Task 3.1: Evaluate Your Improved Algorithm with difficult environment
        # 
        # 
        # ### a) Training
        # - **Random Seeds**: Train your algorithm using three distinct random seeds [0,1,2] to ensure robustness and repeatability.
        # 
        # ### b) Evaluation
        # - **Environment**: Evaluate your algorithm exclusively in the **difficult-level difficulty environment** to focus your improvements.
        # 
        # ### c) Code Compatibility
        # - Ensure that your code is **fully compatible** with all existing functions in other files, maintaining the integrity of the overall project structure.
        # 
        # ---
        # 
        # 
        # from algos.ddpg_agent import DDPGAgent
        # from algos.ppo_agent import PPOAgent
        # from algos.ddpg_extension import DDPGExtension
        # implement your improved algorithm either in algo/ddpg_extension.py or algo/ppo_extension.py

        implemented_algo = 'ddpg_extension'# choose 'ppo_extension' or 'ddpg_extension'
        environment = 'difficult'


        training_seeds = []
        for i in range(3):
            config=setup(algo=implemented_algo, env=environment)
            config["theta"] = theta 
            config["sigma"] = sigma
            config["seed"] = i
            training_seeds.append(i)

            if config["args"].algo_name == 'ppo_extension':
                agent=PPOExtension(config)
            elif config["args"].algo_name == 'ddpg_extension':
                agent=DDPGExtension(config)
            else:
                raise Exception('Please use ppo or ddpg!')

            # Train the agent using selected algorithm    
            agent.train()



        training_seeds = []
        for i in range(3):
            config=setup(algo=implemented_algo, env=environment)
            config["theta"] = theta 
            config["sigma"] = sigma
            config["seed"] = i
            training_seeds.append(i)


            if config["args"].algo_name == 'ppo_extension':
                agent=PPOExtension(config)
            elif config["args"].algo_name == 'ddpg_extension':
                agent=DDPGExtension(config)
            else:
                raise Exception('Please use ppo or ddpg!')

            # Test the agent in the selected environment
            output = test(agent, environment, implemented_algo)
            result_dif["log"].append(output[0])
            result_dif["reward"].append(output[1])
            result_dif["std"].append(output[2])
            result_dif["len"].append(output[3])
            result_dif["seed"].append(i)
            result_dif["theta"].append(theta)
            result_dif["sigma"].append(sigma)

        # **Write your answers here**:
        # 
        # 
        #    
        # - PPO_extension_Difficult_environment:
        #     - mean:
        #     - standard deviation:
        # 
        #  or 
        #  
        #  
        # - DDPG_extension_Difficult_environment:
        #     - mean:
        #     - standard deviation:
        #  
        #  ---

        # ## Task 3.2: Plot the Improved Algorithm's Performance 
        # 
        # #### Display the Plots
        # Display the training performance of your improved algorithm, similar to what was done in Task 2.2.
        # 
        # #### Paths
        # If the code runs successfully, your plot should be saved to the following paths:
        # 
        # - **Improved Difficult**: 
        #   - `results/SandingEnvDifficult/ppo_extension/logging/figure_statistical_SandingEnvDifficult.pdf`
        #   
        #   or
        #   
        #   - `results/SandingEnvDifficult/ddpg_extension/logging/figure_statistical_SandingEnvDifficult.pdf`
        # 

        ## Run the following code to plot PPO or DDPG's training performances
        import warnings
        warnings.filterwarnings('ignore')

        # Uncomment the algorithm you chose 
        implemented_algo = 'ddpg_extension'
        environment = 'difficult'

        # Loop over the three difficulty levels

        training_seeds = [0,1,2]

        config=setup(algo=implemented_algo, env=environment, render=False)
        config["theta"] = theta 
        config["sigma"] = sigma
        config["seed"] = 0

        agent= DDPGExtension(config)# or PPOExtension(config)

        # plot the statistical training curves with specific random seeds
        cu.plot_algorithm_training(agent.logging_dir, training_seeds, agent.env_name, implemented_algo)


        # ## Task 3.3: Plot improved algorithm's and original's comparison performance
        # 
        # ### Display the plots:
        # Display the training performance of your improvement algorithm, similarly as in task 2.3
        # 
        # ### Paths:
        # Your plot should be plotted in the following paths if the code runs successfully:
        # 
        # - **Original vs Improved (difficult environment)**: 
        #   - `results/SandingEnvDifficult/compare_ddpg_ddpg_extension.pdf`
        #   - or 
        #   - `results/SandingEnvDifficult/compare_ppo_ppo_extension.pdf`
        #   

        ## Run the following code to draw the comparison plots of PPO and DDPG's training performances
        import warnings
        warnings.filterwarnings('ignore')

        environment = 'difficult'

        orgin_alo_name = 'ddpg' # or 'ppo'
        improved_alo_name = 'ddpg_extension'# or 'ppo_extension'

        config=setup(algo=orgin_alo_name, env=environment, render=False)
        origin_agent = DDPGAgent(config)# or PPOAgent(config)

        config=setup(algo=improved_alo_name, env=environment, render=False)
        config["theta"] = theta 
        config["sigma"] = sigma
        improved_agent = DDPGExtension(config)# or PPOExtension(config)

        # make the comparison plot
        cu.compare_algorithm_training(origin_agent, improved_agent, seeds=[0,1,2], param = param)

np.save("stats_difficult", result_dif)
np.save("stats_medium", result_med)