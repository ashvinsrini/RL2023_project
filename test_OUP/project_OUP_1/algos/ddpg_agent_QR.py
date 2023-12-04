from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path
import pdb

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGAgent(BaseAgent):
    def __init__(self, config=None):
        super(DDPGAgent, self).__init__(config)
        #pdb.set_trace()
        self.device = "cuda" #self.cfg.device  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.name = 'ddpg'
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        self.max_action = self.cfg.max_action
        self.lr=self.cfg.lr
        
        self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        
        
        self.q = Critic(state_dim, self.action_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))
        
        
        #pdb.set_trace()
        self.buffer =  ReplayBuffer([state_dim], self.action_dim, max_size=int(float(self.cfg.buffer_size)))
        
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000 # collect 5k random data for better exploration
        self.max_episode_steps=self.cfg.max_episode_steps
        self.N_QUANTILES = 32

    def update(self,):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head # update the network once per transition

        if self.buffer_ptr > self.random_transition: # update once we have enough data
            for _ in range(update_iter):
                info = self._update()
        
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr       
        return info


    def _update(self,):
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        # batch contains:
        #    state = batch.state, shape [batch, state_dim]
        #    action = batch.action, shape [batch, action_dim]
        #    next_state = batch.next_state, shape [batch, state_dim]
        #    reward = batch.reward, shape [batch, 1]
        #    not_done = batch.not_done, shape [batch, 1]

        # TODO:
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using u.soft_update_params() (See the DQN code)
        state, action, next_state, reward, not_done = batch.state, batch.action, batch.next_state, batch.reward, batch.not_done
        # compute current q
        current_q = self.q(batch.state, batch.action)
        
        # compute target q
        with torch.no_grad():
            # Target Q value: Q'(s', pi'(s'))
            next_action = self.pi_target(next_state)
            target_q = reward + self.gamma * self.q_target(next_state, next_action)*not_done

        # Compute critic loss using quantile regression loss
        quantiles = torch.tensor(np.linspace(0.0, 1.0, num=self.N_QUANTILES), device=self.device)
        quantile_vals = current_q.unsqueeze(1) + (quantiles.unsqueeze(0) - 0.5) * (target_q.unsqueeze(1) - current_q.unsqueeze(1))
        quantile_loss = F.smooth_l1_loss(quantile_vals, target_q.unsqueeze(1).expand_as(quantile_vals).detach(), reduction='none')
        quantile_loss = (quantile_loss * (quantiles - (quantiles < 0.5).float())).mean()
        
        # # compute critic loss
        # critic_loss = F.mse_loss(current_q, target_q)


        # optimize the critic
        self.q_optim.zero_grad()
        # critic_loss.backward()
        quantile_loss.backward()
        self.q_optim.step()

        # compute actor loss
        actor_loss = -self.q(batch.state, self.pi(batch.state)).mean()

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and target pi using u.soft_update_params() function
        cu.soft_update_params(self.q, self.q_target, self.tau)
        cu.soft_update_params(self.pi, self.pi_target, self.tau)

        ########## Your code ends here. ##########


        return {}

    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        
        if observation.ndim == 1: observation = observation[None] # add the batch dimension
        x = torch.from_numpy(observation).float().to(self.device)

        if (self.buffer_ptr < self.random_transition) and (evaluation == False): # collect random trajectories for better exploration.
            #print('selecting a random action, this should not happen during testing')
            action = torch.rand(self.action_dim)
        else:
            #expl_noise = 0.1 * self.max_action # the stddev of the expl_noise if not evaluation
            expl_noise = 0.3
            # TODO:
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action's shape is correct.
 
            # Use the policy to calculate the action to execute
            action = self.pi(x)
            #pdb.set_trace()
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            if not evaluation:
                action = action + torch.normal(0, expl_noise, size=action.shape).to(self.device)

            # Clip the action to ensure it's within the range
            action = torch.clamp(action, -self.max_action, self.max_action)


            ########## Your code ends here. ##########



            ########## Your code ends here. ##########       
        return action, {} # just return a positional value

    
    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)
        
        
        
    def train_iteration(self):
        #start = time.perf_counter()
        # Run actual training        
        reward_sum, timesteps, done = 0, 0, False
        # Reset the environment and observe the initial state
        obs, _ = self.env.reset()
        while not done:
            
            # Sample action from policy
            action, _ =  self.get_action(obs)
            #pdb.set_trace()
            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _, _ = self.env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)        
            
            done_bool = float(done) if timesteps < self.max_episode_steps else 0 
            self.record(obs, action, next_obs, reward, done_bool)
                
            # Store total episode reward
            reward_sum += reward
            timesteps += 1
            
            if timesteps >= self.max_episode_steps:
                done = True
            # update observation
            obs = next_obs.copy()

        # update the policy after one episode
        #s = time.perf_counter()
        info = self.update()
        #e = time.perf_counter()
        
        # Return stats of training
        info.update({
                    'episode_length': timesteps,
                    'ep_reward': reward_sum,
                    })
        
        end = time.perf_counter()
        return info
        
    def train(self):
        if self.cfg.save_logging:
            L = cu.Logger() # create a simple logger to record stats
        start = time.perf_counter()
        total_step=0
        run_episode_reward=[]
        log_count=0
        
        for ep in range(self.cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = self.train_iteration()
            train_info.update({'episodes': ep})
            total_step+=train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            
            if total_step>self.cfg.log_interval*log_count:
                average_return=sum(run_episode_reward)/len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step} finished. Average episode return: {average_return}")
                if self.cfg.save_logging:
                    train_info.update({'average_return':average_return})
                    L.log(**train_info)
                run_episode_reward=[]
                log_count+=1

        if self.cfg.save_model:
            self.save_model()
            
        logging_path = str(self.logging_dir)+'/logs'   
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()

        end = time.perf_counter()
        train_time = (end-start)/60
        print('------ Training Finished ------')
        print(f'Total traning time is {train_time}mins')
        
    def load_model(self):
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        d = torch.load(filepath)
        self.q.load_state_dict(d['q'])
        self.q_target.load_state_dict(d['q_target'])
        self.pi.load_state_dict(d['pi'])
        self.pi_target.load_state_dict(d['pi_target'])
    
    def save_model(self):   
        # define the save path, do not modify
        filepath=str(self.model_dir)+'/model_parameters_'+str(self.seed)+'.pt'
        
        torch.save({
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'pi': self.pi.state_dict(),
            'pi_target': self.pi_target.state_dict()
        }, filepath)
        print("Saved model to", filepath, "...")
        
        
