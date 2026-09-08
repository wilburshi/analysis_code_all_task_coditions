#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time


# In[ ]:


# --- Check for GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


data_save_path = '/gpfs/marilyn/pi/nandy/Weikang/Effort_based_cooperation_more_results/' +                  'MARL_data/training_withGAE_PPO_fixedPullCoolTime_discreteEffort/'
#
fig_save_path = '/gpfs/marilyn/pi/nandy/Weikang/Effort_based_cooperation_more_results/' +                  'MARL_figures/training_withGAE_PPO_fixedPullCoolTime_discreteEffort/'


# ### Agent brain (unchanged architecture from the gaze-cooperation model)\nOnly the *input size* changes (public input grows from 5 to 7 to carry self/partner effort cost). Hidden size, heads, and everything else stays identical, which is what makes the weight-transfer loader below possible.

# In[ ]:


#################################################################
# 1. AGENT BRAIN CLASS (unchanged from gaze-cooperation model)
#################################################################

class AgentBrain(nn.Module):
    """
    An Actor-Critic agent with a shared RNN trunk (GRU).
    Identical to the gaze-cooperation model -- only input_size changes
    at construction time (7 public + 5 private = 12, vs. 5+5=10 before).
    """
    def __init__(self, public_input_size, private_input_size, rnn_hidden_size,
                 move_action_size, pull_action_size, gaze_action_size):

        super(AgentBrain, self).__init__()

        self.input_size = public_input_size + private_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.move_action_size = move_action_size
        self.pull_action_size = pull_action_size
        self.gaze_action_size = gaze_action_size

        # --- 1. The Shared RNN "Trunk" ---
        self.rnn_trunk = nn.GRUCell(self.input_size, self.rnn_hidden_size)

        # --- 2. The "Actor" Head ---
        self.move_head = nn.Linear(self.rnn_hidden_size, move_action_size)
        self.pull_head = nn.Linear(self.rnn_hidden_size, pull_action_size)
        self.gaze_head = nn.Linear(self.rnn_hidden_size, gaze_action_size)

        # --- 3. The "Critic" Head ---
        self.critic_head = nn.Linear(self.rnn_hidden_size, 1)

    def forward(self, public_input, private_gaze_input, prev_hidden_state, pull_mask):

        full_input = torch.cat([public_input, private_gaze_input], dim=-1)
        current_hidden_state = self.rnn_trunk(full_input, prev_hidden_state)

        value = self.critic_head(current_hidden_state)

        move_logits = self.move_head(current_hidden_state)
        gaze_logits = self.gaze_head(current_hidden_state)

        pull_logits = self.pull_head(current_hidden_state)
        pull_logits = pull_logits + pull_mask

        return (move_logits, pull_logits, gaze_logits), value, current_hidden_state


# ### environment: effort-inequity cooperation task
# 
# #### Two-phase design: one agent stays fixed at FORCE_LOW (100g) while the other's force is resampled every block_len steps from a discrete set of levels — 100, 200, ..., 900, matching your pilot task's actual increments rather than continuous sampling. No automatic session-flipping; which agent is fixed is set explicitly and only changes when you call set_fixed_agent() (e.g., at the Phase 1 → Phase 2 boundary), so you control the A-then-B / B-then-A ordering directly in the training loop rather than leaving it to chance.
# 
# #### block_len=1500: force changes every 1500 steps within a phase
# #### DISCRETE_FORCE_LEVELS = np.arange(100, 901, 100): matches your actual behavioral task's force increments
# #### Force cost is still modeled as absolute lever force, linearly mapped to RL cost (unchanged from before) — your Fig.4 analysis's relative/previous-block-difference framing is still a possible swap in _set_new_block() later if you want to match that specific analysis

# In[ ]:


#################################################################
# 2. EFFORT-INEQUITY ENVIRONMENT (discrete force levels, two-phase design)
#################################################################

# --- Force -> RL cost mapping ---
FORCE_LOW = 100          # grams -- the fixed agent's constant force
DISCRETE_FORCE_LEVELS = np.arange(100, 901, 100)   # [100, 200, ..., 900] -- matches pilot task increments
FORCE_TO_COST_SCALE = 0.3 / 900    # so 900g ~ comparable magnitude to old EFFORT_HIGH=0.3

def force_to_cost(force_grams):
    return force_grams * FORCE_TO_COST_SCALE


class CurriculumEnv:
    def __init__(self, block_len=1500, fixed_agent='B'):
        self.pull_reset_time = 20
        self.training_phase = "SELF_REWARD"
        self.current_sync_window = 30

        # --- Gaze cost (kept from the gaze-cooperation model) ---
        self.gaze_cost = 0.0

        # --- Effort-inequity: discrete force levels, no session structure ---
        self.block_len = block_len
        self._effort_step_counter = 0
        self.fixed_agent = fixed_agent

        self.reset()
        self._set_new_block()

    def set_training_phase(self, phase, sync_window=30):
        self.training_phase = phase
        self.current_sync_window = sync_window
        print(f"*** ENV UPDATED: Phase={self.training_phase}, "
              f"SyncWindow={self.current_sync_window}, GazeCost={self.gaze_cost} ***")

    def set_fixed_agent(self, agent):
        """
        Switches which agent is fixed (e.g. at a Phase 1 -> Phase 2 boundary:
        env.set_fixed_agent('A') switches from B-varying to A-varying).
        Immediately resamples a new block so the switch takes effect right away.
        """
        assert agent in ('A', 'B'), "agent must be 'A' or 'B'"
        self.fixed_agent = agent
        self._set_new_block()
        print(f"*** FIXED AGENT SWITCHED: now {agent} is fixed at {FORCE_LOW}g, "
              f"{'B' if agent == 'A' else 'A'} is varying ***")

    def _set_new_block(self):
        """
        Advances one block: resamples the varying agent's force from the
        DISCRETE set of levels. Which agent is fixed does NOT change here --
        only set_fixed_agent() changes that, explicitly, under your control.
        """
        varying_force = np.random.choice(DISCRETE_FORCE_LEVELS)
        if self.fixed_agent == 'A':
            self.force_A, self.force_B = FORCE_LOW, varying_force
        else:
            self.force_A, self.force_B = varying_force, FORCE_LOW

        self.cost_A = force_to_cost(self.force_A)
        self.cost_B = force_to_cost(self.force_B)
        self.cond_label = f"fixed{self.fixed_agent}_A{self.force_A:.0f}g_B{self.force_B:.0f}g"

    def _get_state(self):
        on_cooldown_A = self.agent_A_pull_cooldown > 0
        is_drinking_A = self.A_is_drinking
        can_pull_A = not (on_cooldown_A or is_drinking_A)

        on_cooldown_B = self.agent_B_pull_cooldown > 0
        is_drinking_B = self.B_is_drinking
        can_pull_B = not (on_cooldown_B or is_drinking_B)

        state_A = {
            'can_pull': can_pull_A,
            'self_is_drinking': self.A_is_drinking,
            'self_on_cooldown': on_cooldown_A,
            'self_cooldown_remaining': self.agent_A_pull_cooldown,
            'self_drink_timer_remaining': self.A_drink_timer,
            'success_sound': self.success_sound_played
        }
        state_B = {
            'can_pull': can_pull_B,
            'self_is_drinking': self.B_is_drinking,
            'self_on_cooldown': on_cooldown_B,
            'self_cooldown_remaining': self.agent_B_pull_cooldown,
            'self_drink_timer_remaining': self.B_drink_timer,
            'success_sound': self.success_sound_played
        }
        return state_A, state_B

    def reset(self):
        self.current_step = 0
        self.pull_A_timestamp = -1
        self.pull_B_timestamp = -1
        self.agent_A_pull_cooldown = 0
        self.agent_B_pull_cooldown = 0
        self.success_sound_played = 0
        self.A_time_since_pull = 1000
        self.B_time_since_pull = 1000
        self.A_is_drinking = False
        self.B_is_drinking = False
        self.A_drink_timer = 0
        self.B_drink_timer = 0
        return self._get_state()

    def step(self, action_A, action_B):
        self.current_step += 1
        reward_A, reward_B = -0.01, -0.01  # base cost of living
        self.success_sound_played = 0
        done = False
        coordination_attempt_complete = False

        self.A_time_since_pull += 1
        self.B_time_since_pull += 1

        # --- Gaze cost (unchanged from gaze-cooperation model) ---
        if self.training_phase != "SELF_REWARD":
            if action_A['gaze'] == 1:
                reward_A -= self.gaze_cost
            if action_B['gaze'] == 1:
                reward_B -= self.gaze_cost

        # --- Standard timer logic ---
        on_cooldown_A = self.agent_A_pull_cooldown > 0
        is_drinking_A = self.A_is_drinking
        can_pull_A = not (on_cooldown_A or is_drinking_A)

        on_cooldown_B = self.agent_B_pull_cooldown > 0
        is_drinking_B = self.B_is_drinking
        can_pull_B = not (on_cooldown_B or is_drinking_B)

        if self.A_drink_timer > 0:
            self.A_drink_timer -= 1
            if self.A_drink_timer == 0: self.A_is_drinking = False
        if self.B_drink_timer > 0:
            self.B_drink_timer -= 1
            if self.B_drink_timer == 0: self.B_is_drinking = False
        if self.agent_A_pull_cooldown > 0: self.agent_A_pull_cooldown -= 1
        if self.agent_B_pull_cooldown > 0: self.agent_B_pull_cooldown -= 1

        # --- Effort cost on ATTEMPTED pulls (mask-permitting) ---
        if action_A['pull'] == 1 and can_pull_A:
            reward_A -= self.cost_A
        if action_B['pull'] == 1 and can_pull_B:
            reward_B -= self.cost_B

        if self.training_phase == "SELF_REWARD":
            if (action_A['pull'] == 1 and can_pull_A):
                self.A_time_since_pull = 0; reward_A += 15.0; self.success_sound_played = 1
                self.agent_A_pull_cooldown = self.pull_reset_time
                self.A_is_drinking = True; self.A_drink_timer = np.random.randint(30, 50)
            if (action_B['pull'] == 1 and can_pull_B):
                self.B_time_since_pull = 0; reward_B += 15.0; self.success_sound_played = 1
                self.agent_B_pull_cooldown = self.pull_reset_time
                self.B_is_drinking = True; self.B_drink_timer = np.random.randint(30, 50)

        else:  # COOP_ASSISTED or COOP_FULL
            if (action_A['pull'] == 1 and can_pull_A):
                self.A_time_since_pull = 0
                self.pull_A_timestamp = self.current_step
                self.agent_A_pull_cooldown = self.pull_reset_time
                if self.pull_B_timestamp != -1: coordination_attempt_complete = True

            if (action_B['pull'] == 1 and can_pull_B):
                self.B_time_since_pull = 0
                self.pull_B_timestamp = self.current_step
                self.agent_B_pull_cooldown = self.pull_reset_time
                if self.pull_A_timestamp != -1: coordination_attempt_complete = True

            if coordination_attempt_complete:
                if abs(self.pull_A_timestamp - self.pull_B_timestamp) <= self.current_sync_window:
                    reward_A += 15.0; reward_B += 15.0
                    self.success_sound_played = 1
                    self.A_is_drinking = True; self.A_drink_timer = np.random.randint(30, 50)
                    self.B_is_drinking = True; self.B_drink_timer = np.random.randint(30, 50)
                else:
                    if self.training_phase == "COOP_ASSISTED": reward_A += 5.0; reward_B += 5.0
                    else: reward_A -= 0.5; reward_B -= 0.5
                self.pull_A_timestamp = -1
                self.pull_B_timestamp = -1

        # --- Advance effort block ---
        self._effort_step_counter += 1
        if self._effort_step_counter % self.block_len == 0:
            self._set_new_block()

        next_state_A, next_state_B = self._get_state()
        return next_state_A, next_state_B, reward_A, reward_B, done


# ### helper functions\n`get_private_state`, `calculate_gae`, `sample_actions` are unchanged from the gaze-cooperation model. `preprocess_state` is replaced with `preprocess_state_with_effort`, which appends normalized self/partner effort cost to the public vector.

# In[ ]:


#################################################################
# 3. HELPER FUNCTIONS
#################################################################

def preprocess_state_with_effort(state_dict, self_force, partner_force):
    """
    Public input, now size 7: the original 5 dims + normalized self_force + partner_force.

    Takes FORCE (grams), not cost, and normalizes it directly to [0, 1] via
    (force - FORCE_LOW) / (DISCRETE_FORCE_LEVELS.max() - FORCE_LOW). The
    fixed agent (always FORCE_LOW=100g) gets exactly 0.0; the varying agent
    gets one of 9 discrete values from 0.0 (100g) to 1.0 (900g).

    BUG HISTORY: earlier version passed COST and binarized it
    (`1.0 if cost > 0 else 0.0`) -- that made sense for the old discrete
    EFFORT_LOW=0.0/EFFORT_HIGH=0.3 design, but broke silently once
    FORCE_LOW became 100g (never zero): self_cost was ALWAYS > 0, so the
    network received a constant 1.0 regardless of actual force, and had
    NO information to condition behavior on. This fix restores that signal.
    """
    is_drinking = torch.tensor([1.0 if state_dict['self_is_drinking'] else 0.0])
    can_pull = torch.tensor([1.0 if state_dict['can_pull'] else 0.0])
    success_sound = torch.tensor([1.0 if state_dict['success_sound'] else 0.0])
    drink_timer = torch.tensor([state_dict['self_drink_timer_remaining'] / 50.0])
    cooldown_timer = torch.tensor([state_dict['self_cooldown_remaining'] / 20.0])

    force_range = DISCRETE_FORCE_LEVELS.max() - FORCE_LOW  # = 800
    self_force_norm = torch.tensor([np.clip((self_force - FORCE_LOW) / force_range, 0.0, 1.0)])
    partner_force_norm = torch.tensor([np.clip((partner_force - FORCE_LOW) / force_range, 0.0, 1.0)])

    tensor_state = torch.cat([
        is_drinking, can_pull, success_sound, drink_timer, cooldown_timer,
        self_force_norm, partner_force_norm
    ], dim=0)

    return tensor_state.float().unsqueeze(0).to(device)

def get_private_state(partner_state_dict, partner_time, sync_window, reliability_level, is_gazing):
    """
    Unchanged from the gaze-cooperation model. Gaze-gated noisy readout of
    partner state. Reliability is held FIXED (not swept) during effort
    training -- see hyperparameters cell -- to avoid conflating two sources
    of task difficulty at once.
    """
    if partner_state_dict['can_pull']:
        true_mean = 1.0
    else:
        true_mean = 0.0

    base_noise = 0.6
    current_sigma = base_noise / (reliability_level + 1e-2)

    if is_gazing:
        observed_value = np.random.normal(loc=true_mean, scale=current_sigma)
        valid_bit = 1.0
    else:
        observed_value = 0.0
        valid_bit = 0.0

    p_signal = torch.tensor([observed_value])
    p_time = torch.tensor([min(partner_time / sync_window, 1.0)])
    p_drink = torch.tensor([1.0 if partner_state_dict['self_is_drinking'] else 0.0])
    p_cool = torch.tensor([1.0 if partner_state_dict['self_on_cooldown'] else 0.0])
    v_bit = torch.tensor([valid_bit])

    tensor_state = torch.cat([p_signal, p_time, p_drink, p_cool, v_bit], dim=0)
    return tensor_state.float().unsqueeze(0).to(device)


def calculate_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    gae = 0
    returns = []
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lambda_ * gae
        returns.insert(0, gae + values[step])
    return torch.cat(returns).view(-1).float().to(device)


def sample_actions(logits_tuple):
    move_logits, pull_logits, gaze_logits = logits_tuple

    move_dist = Categorical(logits=move_logits)
    move_action = move_dist.sample()
    move_log_prob = move_dist.log_prob(move_action)

    pull_dist = Categorical(logits=pull_logits)
    pull_action = pull_dist.sample()
    pull_log_prob = pull_dist.log_prob(pull_action)

    gaze_dist = Categorical(logits=gaze_logits)
    gaze_action = gaze_dist.sample()
    gaze_log_prob = gaze_dist.log_prob(gaze_action)

    total_log_prob = move_log_prob + pull_log_prob + gaze_log_prob

    actions_dict = {
        'move': move_action.item(),
        'pull': pull_action.item(),
        'gaze': gaze_action.item()
    }
    return actions_dict, total_log_prob


# ### weight transfer: load the gaze-cooperation generalist checkpoint into the wider effort model\nOnly `rnn_trunk.weight_ih` changes shape (input size 10 -> 12). Everything else (recurrent weights, biases, all four output heads) transfers directly since those only depend on `hidden_size`, which is unchanged.\n\n**Path note:** this notebook lives in `MARL_effort_based_cooperation/`, while the source checkpoints live in `MARL_gaze_accumulation_strategies/training_withGAE_PPO_fixedPullCoolTime/`. The path below is a **relative** reference between those two sibling folders -- adjust `OLD_CHECKPOINT_DIR` if your actual folder depth differs, or point it at a local copy if you'd rather not depend on the sibling folder (safer for sbatch jobs, since working directory at submission time isn't always what you expect).

# In[ ]:


#################################################################
# 4. WEIGHT TRANSFER
#################################################################

OLD_CHECKPOINT_DIR = "/gpfs/radev/home/ws523/marmoset_tracking_DLCv2/following_up_analysis"+"/MARL_gaze_accumulation_strategies/training_withGAE_PPO_fixedPullCoolTime"
OLD_PUBLIC_INPUT_SIZE = 5
OLD_PRIVATE_INPUT_SIZE = 5

def load_and_expand_agent(new_model, old_checkpoint_path,
                           old_public_size=OLD_PUBLIC_INPUT_SIZE,
                           old_private_size=OLD_PRIVATE_INPUT_SIZE,
                           new_public_size=7,
                           new_private_size=5):
    old_state = torch.load(old_checkpoint_path, map_location='cpu')
    new_state = new_model.state_dict()

    for key, old_tensor in old_state.items():
        if key == 'rnn_trunk.weight_ih':
            new_tensor = new_state[key].clone()
            new_tensor[:, :old_public_size] = old_tensor[:, :old_public_size]
            # ZERO-init the new force columns (previously random-init) -- with
            # random init, force*random_weight injects noise into the recurrent
            # hidden state from step 0, before any training, which can knock an
            # otherwise-balanced policy into a degenerate regime immediately upon
            # transfer. Zero-init means these columns contribute nothing until the
            # network actually learns a real relationship through gradient updates.
            new_tensor[:, old_public_size:new_public_size] = 0.0
            new_tensor[:, new_public_size:new_public_size + old_private_size] =                 old_tensor[:, old_public_size:old_public_size + old_private_size]
            new_state[key] = new_tensor
            print(f"Expanded '{key}': {tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)} "
                  f"(effort-cost columns ZERO-initialized)")
        else:
            new_state[key] = old_tensor

    new_model.load_state_dict(new_state)
    print(f"Loaded and expanded weights from {old_checkpoint_path}")
    return new_model


# ### training hyperparameters

# In[ ]:


#################################################################
# 5. TRAINING HYPERPARAMETERS
#################################################################

RNN_HIDDEN_SIZE = 128
PUBLIC_INPUT_SIZE = 7    # 5 original + self_force + partner_force
PRIVATE_INPUT_SIZE = 5   # unchanged

MOVE_ACTIONS = 5
PULL_ACTIONS = 2
GAZE_ACTIONS = 2

TARGET_WINDOW = 10        # keep the "1s" sync window from the trained base
FIXED_RELIABILITY = 1.0   # held fixed during effort training -- see note above
BLOCK_LEN = 1500           # steps per force block -- matches your two-phase design

LR_FINE_TUNE = 5e-5       # matches your other continue_training_* fine-tunes
GAMMA = 0.99
N_STEPS = 500
CONTINUE_STEPS = 100000   # per phase (100k A-varying + 100k B-varying = 200k total per run)
ENTROPY_COEF = 0.0001
VALUE_LOSS_COEF = 0.1
EPS_CLIP = 0.2

GAZE_COST = 0.2   # carried over from the efficient/generalist gaze model


# ### two-phase training: A-then-B / B-then-A
# 
# For each `(order, seed)` combination, trains through two sequential 100k-step phases on the fixed-cooldown environment at a fixed cost-scale multiplier (18x): one phase with Agent A varying force (100-900g, discrete levels) while B stays fixed at 100g, the other phase with roles reversed. `order='A_then_B'` runs A-varying first; `order='B_then_A'` runs B-varying first -- running both orderings lets you check whether training order itself affects the result, since Phase 2 always starts from Phase-1-shaped weights, not a fresh checkpoint.
# 
# **What gets saved, per `(order, seed)`:**
# - 4 checkpoints: `agent_A/B_{order}_phase1_seed{seed}.pth`, `agent_A/B_{order}_phase2_seed{seed}.pth` -- end-of-phase snapshots only, no intermediate saves
# - 1 combined behavioral log: `training_log_{order}_seed{seed}.pkl`, with a `phase` column (`'phase1'`/`'phase2'`) so you can filter and analyze each phase separately using the same force/pull/gaze/IPI/success-rate code already built for the other sweeps
# 
# **Caveat worth remembering when interpreting Phase 2:** it isn't a fresh test -- it starts from whatever Phase 1 left behind, including the general "warm-up" drift we've seen happen even with force held constant. If Phase 2's force-sensitivity looks different from Phase 1's, that could be a real order effect, or just Phase 2 inheriting an already-partially-trained policy. Comparing A-then-B's Phase 2 against B-then-A's Phase 1 (same role, different training history) is the cleanest way to isolate order effects specifically.
# 
# **Retraining:** each `(order, seed)` combination is skipped automatically if all 4 output files already exist. Set `RETRAIN = 1` in the driver cell to force a full redo.
# 

# In[ ]:


#################################################################
# 6. TWO-PHASE TRAINING (A-then-B / B-then-A, fixed-cooldown)
#################################################################

BASE_SCALE = 0.3 / 900
COST_SCALE_MULTIPLIER = 18.0

CHECKPOINT_SET = "generalist_withGAE_PPO"   # or "efficient_expert" -- pick your starting checkpoint


def run_two_phase_training(order, seed=0, phase_steps=CONTINUE_STEPS, block_len=BLOCK_LEN,
                             cost_multiplier=COST_SCALE_MULTIPLIER, force_retrain=False):
    """
    order: 'A_then_B' -> Phase 1: A varying / B fixed, Phase 2: B varying / A fixed
           'B_then_A' -> Phase 1: B varying / A fixed, Phase 2: A varying / B fixed

    Saves agent_A/B weights at the END of EACH phase (4 .pth files per run),
    plus one combined behavioral log (tagged by 'phase') so you can check
    performance at each checkpoint separately.
    """
    assert order in ('A_then_B', 'B_then_A')
    phase1_fixed = 'B' if order == 'A_then_B' else 'A'
    phase2_fixed = 'A' if order == 'A_then_B' else 'B'

    out_files = [
        data_save_path + f"agent_A_{order}_{CHECKPOINT_SET}_phase1_seed{seed}.pth",
        data_save_path + f"agent_B_{order}_{CHECKPOINT_SET}_phase1_seed{seed}.pth",
        data_save_path + f"agent_A_{order}_{CHECKPOINT_SET}_phase2_seed{seed}.pth",
        data_save_path + f"agent_B_{order}_{CHECKPOINT_SET}_phase2_seed{seed}.pth",
    ]
    if not force_retrain and all(os.path.exists(f) for f in out_files):
        print(f"Found existing outputs for order={order}, seed={seed} -- skipping. "
              f"Call force_retrain=True to redo.")
        return

    np.random.seed(seed)
    torch.manual_seed(seed)

    global FORCE_TO_COST_SCALE
    FORCE_TO_COST_SCALE = BASE_SCALE * cost_multiplier

    env = CurriculumEnv(block_len=block_len, fixed_agent=phase1_fixed)
    env.set_training_phase("COOP_FULL", sync_window=TARGET_WINDOW)
    env.gaze_cost = GAZE_COST

    agent_A = AgentBrain(PUBLIC_INPUT_SIZE, PRIVATE_INPUT_SIZE, RNN_HIDDEN_SIZE,
                         MOVE_ACTIONS, PULL_ACTIONS, GAZE_ACTIONS).to(device)
    agent_B = AgentBrain(PUBLIC_INPUT_SIZE, PRIVATE_INPUT_SIZE, RNN_HIDDEN_SIZE,
                         MOVE_ACTIONS, PULL_ACTIONS, GAZE_ACTIONS).to(device)
    agent_A = load_and_expand_agent(agent_A, f"{OLD_CHECKPOINT_DIR}/agent_A_{CHECKPOINT_SET}.pth")
    agent_B = load_and_expand_agent(agent_B, f"{OLD_CHECKPOINT_DIR}/agent_B_{CHECKPOINT_SET}.pth")

    optimizer_A = optim.Adam(agent_A.parameters(), lr=LR_FINE_TUNE)
    optimizer_B = optim.Adam(agent_B.parameters(), lr=LR_FINE_TUNE)

    state_A_dict, state_B_dict = env.reset()
    public_A = preprocess_state_with_effort(state_A_dict, env.force_A, env.force_B)
    public_B = preprocess_state_with_effort(state_B_dict, env.force_B, env.force_A)
    private_A = torch.zeros(1, PRIVATE_INPUT_SIZE).to(device)
    private_B = torch.zeros(1, PRIVATE_INPUT_SIZE).to(device)
    hidden_A = torch.zeros(1, RNN_HIDDEN_SIZE).to(device)
    hidden_B = torch.zeros(1, RNN_HIDDEN_SIZE).to(device)

    full_log = []

    def run_phase(n_steps, phase_label):
        nonlocal public_A, public_B, private_A, private_B, hidden_A, hidden_B, state_A_dict, state_B_dict

        for update_step in range(n_steps // N_STEPS):
            mem_pub_A, mem_priv_A, mem_mask_A = [], [], []
            mem_act_A, mem_val_A, mem_rew_A, mem_logp_A = [], [], [], []
            mem_pub_B, mem_priv_B, mem_mask_B = [], [], []
            mem_act_B, mem_val_B, mem_rew_B, mem_logp_B = [], [], [], []

            start_hidden_A, start_hidden_B = hidden_A.detach(), hidden_B.detach()
            hidden_A, hidden_B = start_hidden_A, start_hidden_B

            for _ in range(N_STEPS):
                mask_A = torch.tensor([[0., 0. if state_A_dict['can_pull'] else -1e9]], device=device).float()
                mask_B = torch.tensor([[0., 0. if state_B_dict['can_pull'] else -1e9]], device=device).float()

                mem_pub_A.append(public_A); mem_priv_A.append(private_A); mem_mask_A.append(mask_A)
                mem_pub_B.append(public_B); mem_priv_B.append(private_B); mem_mask_B.append(mask_B)

                with torch.no_grad():
                    logits_A, value_A, hidden_A = agent_A(public_A, private_A, hidden_A, mask_A)
                    logits_B, value_B, hidden_B = agent_B(public_B, private_B, hidden_B, mask_B)

                act_A, log_prob_A = sample_actions(logits_A)
                act_B, log_prob_B = sample_actions(logits_B)
                mem_act_A.append(act_A); mem_val_A.append(value_A); mem_logp_A.append(log_prob_A)
                mem_act_B.append(act_B); mem_val_B.append(value_B); mem_logp_B.append(log_prob_B)

                force_A_now, force_B_now = env.force_A, env.force_B
                can_pull_A_now, can_pull_B_now = state_A_dict['can_pull'], state_B_dict['can_pull']
                next_A, next_B, r_A, r_B, done = env.step(act_A, act_B)
                mem_rew_A.append(r_A); mem_rew_B.append(r_B)

                full_log.append({'phase': phase_label, 'force_A': force_A_now, 'force_B': force_B_now,
                                  'A_pull': act_A['pull'], 'B_pull': act_B['pull'],
                                  'A_gaze': act_A['gaze'], 'B_gaze': act_B['gaze'],
                                  'can_pull_A': can_pull_A_now, 'can_pull_B': can_pull_B_now})

                private_A = get_private_state(next_B, env.B_time_since_pull, env.current_sync_window, FIXED_RELIABILITY, act_A['gaze']==1)
                private_B = get_private_state(next_A, env.A_time_since_pull, env.current_sync_window, FIXED_RELIABILITY, act_B['gaze']==1)
                public_A = preprocess_state_with_effort(next_A, env.force_A, env.force_B)
                public_B = preprocess_state_with_effort(next_B, env.force_B, env.force_A)
                state_A_dict, state_B_dict = next_A, next_B

            with torch.no_grad():
                _, next_value_A, _ = agent_A(public_A, private_A, hidden_A, mask_A)
                _, next_value_B, _ = agent_B(public_B, private_B, hidden_B, mask_B)
            returns_A = calculate_gae(mem_rew_A, mem_val_A, next_value_A)
            returns_B = calculate_gae(mem_rew_B, mem_val_B, next_value_B)
            values_A_old = torch.cat(mem_val_A).view(-1)
            values_B_old = torch.cat(mem_val_B).view(-1)
            adv_A = (returns_A - values_A_old); adv_A = (adv_A - adv_A.mean()) / (adv_A.std() + 1e-8)
            adv_B = (returns_B - values_B_old); adv_B = (adv_B - adv_B.mean()) / (adv_B.std() + 1e-8)

            curr_hidden_A, curr_hidden_B = start_hidden_A, start_hidden_B
            new_lp_A_list, new_lp_B_list = [], []
            entropy_A_list, entropy_B_list = [], []
            for t in range(N_STEPS):
                (l_A), _, curr_hidden_A = agent_A(mem_pub_A[t], mem_priv_A[t], curr_hidden_A, mem_mask_A[t])
                m,p,g = l_A; a = mem_act_A[t]
                new_lp_A_list.append(Categorical(logits=m).log_prob(torch.tensor(a['move']).to(device)) +
                                      Categorical(logits=p).log_prob(torch.tensor(a['pull']).to(device)) +
                                      Categorical(logits=g).log_prob(torch.tensor(a['gaze']).to(device)))
                ent_A = -(F.softmax(p, -1)*F.log_softmax(p, -1)).sum()
                entropy_A_list.append(ent_A)

                (l_B), _, curr_hidden_B = agent_B(mem_pub_B[t], mem_priv_B[t], curr_hidden_B, mem_mask_B[t])
                m,p,g = l_B; a = mem_act_B[t]
                new_lp_B_list.append(Categorical(logits=m).log_prob(torch.tensor(a['move']).to(device)) +
                                      Categorical(logits=p).log_prob(torch.tensor(a['pull']).to(device)) +
                                      Categorical(logits=g).log_prob(torch.tensor(a['gaze']).to(device)))
                ent_B = -(F.softmax(p, -1)*F.log_softmax(p, -1)).sum()
                entropy_B_list.append(ent_B)

            new_lp_A = torch.stack(new_lp_A_list).view(-1); old_lp_A = torch.cat(mem_logp_A).detach().view(-1)
            new_lp_B = torch.stack(new_lp_B_list).view(-1); old_lp_B = torch.cat(mem_logp_B).detach().view(-1)
            mean_ent_A = torch.stack(entropy_A_list).mean()
            mean_ent_B = torch.stack(entropy_B_list).mean()

            ratio_A = torch.exp(new_lp_A - old_lp_A)
            loss_A = -torch.min(ratio_A*adv_A, torch.clamp(ratio_A,1-EPS_CLIP,1+EPS_CLIP)*adv_A).mean() +                      VALUE_LOSS_COEF*F.mse_loss(values_A_old, returns_A) - ENTROPY_COEF * mean_ent_A
            ratio_B = torch.exp(new_lp_B - old_lp_B)
            loss_B = -torch.min(ratio_B*adv_B, torch.clamp(ratio_B,1-EPS_CLIP,1+EPS_CLIP)*adv_B).mean() +                      VALUE_LOSS_COEF*F.mse_loss(values_B_old, returns_B) - ENTROPY_COEF * mean_ent_B

            optimizer_A.zero_grad(); loss_A.backward(); torch.nn.utils.clip_grad_norm_(agent_A.parameters(), 0.5); optimizer_A.step()
            optimizer_B.zero_grad(); loss_B.backward(); torch.nn.utils.clip_grad_norm_(agent_B.parameters(), 0.5); optimizer_B.step()

            if update_step % (5000 // N_STEPS) == 0:
                steps_so_far = update_step * N_STEPS
                print(f"    [{order} seed={seed} {phase_label}] step {steps_so_far}/{n_steps}, "
                      f"R_A: {sum(mem_rew_A):.2f}, R_B: {sum(mem_rew_B):.2f}")

    # --- Phase 1 ---
    print(f"\n=== {order}, seed={seed}: PHASE 1 (fixed={phase1_fixed}) ===")
    run_phase(phase_steps, 'phase1')
    torch.save(agent_A.state_dict(), data_save_path+f"agent_A_{order}_{CHECKPOINT_SET}_phase1_seed{seed}.pth")
    torch.save(agent_B.state_dict(), data_save_path+f"agent_B_{order}_{CHECKPOINT_SET}_phase1_seed{seed}.pth")
    print("Saved phase-1 checkpoints.")

    # --- Switch to Phase 2 ---
    env.set_fixed_agent(phase2_fixed)

    # --- Phase 2 ---
    print(f"\n=== {order}, seed={seed}: PHASE 2 (fixed={phase2_fixed}) ===")
    run_phase(phase_steps, 'phase2')
    torch.save(agent_A.state_dict(), data_save_path+f"agent_A_{order}_{CHECKPOINT_SET}_phase2_seed{seed}.pth")
    torch.save(agent_B.state_dict(), data_save_path+f"agent_B_{order}_{CHECKPOINT_SET}_phase2_seed{seed}.pth")
    print("Saved phase-2 checkpoints.")

    log_df = pd.DataFrame(full_log)
    log_df.to_pickle(data_save_path+f"training_log_{order}_{CHECKPOINT_SET}_seed{seed}.pkl")
    print(f"Saved training_log_{order}_{CHECKPOINT_SET}_seed{seed}.pkl ({len(log_df)} steps)\n")


# --- Driver: both orderings x however many seeds ---
TRAINING_SEEDS = (0, 1, 2, 3, 4, 5)
RETRAIN = 0   # set to 1 to force-redo everything; 0 = skip any (order, seed) already saved

for order in ['A_then_B', 'B_then_A']:
    for seed in TRAINING_SEEDS:
        run_two_phase_training(order=order, seed=seed, force_retrain=bool(RETRAIN))
        
        


# In[ ]:


#################################################################
# SANITY CHECK: pull rate and gaze rate over TRAINING STEPS
# (not force), per seed, for both agents -- checks whether the
# policy moved at all during training, or was flat from step 0.
#################################################################
if 0:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    CHECKPOINT_SET = "generalist_withGAE_PPO"
    ORDERS = ['A_then_B', 'B_then_A']
    SEEDS = (0, 1, 2, 3, 4, 5)
    STEP_BIN_SIZE = 5000

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for col, phase in enumerate(['phase1', 'phase2']):
        for order in ORDERS:
            for seed in SEEDS:
                fname = data_save_path+f"training_log_{order}_{CHECKPOINT_SET}_seed{seed}.pkl"
                try:
                    log_df = pd.read_pickle(fname)
                except FileNotFoundError:
                    continue

                phase_df = log_df[log_df['phase'] == phase].reset_index(drop=True)
                if len(phase_df) == 0:
                    continue
                phase_df['step_bin'] = (phase_df.index // STEP_BIN_SIZE) * STEP_BIN_SIZE

                # --- Pull rate, conditioned on eligibility, both agents ---
                elig_A = phase_df[phase_df['can_pull_A']]
                elig_B = phase_df[phase_df['can_pull_B']]
                curve_A_pull = elig_A.groupby('step_bin')['A_pull'].mean()
                curve_B_pull = elig_B.groupby('step_bin')['B_pull'].mean()

                axes[0, col].plot(curve_A_pull.index, curve_A_pull.values, '-', color='#2E86AB', alpha=0.35, linewidth=1)
                axes[0, col].plot(curve_B_pull.index, curve_B_pull.values, '-', color='#C73E1D', alpha=0.35, linewidth=1)

                # --- Gaze rate, unconditioned, both agents ---
                curve_A_gaze = phase_df.groupby('step_bin')['A_gaze'].mean()
                curve_B_gaze = phase_df.groupby('step_bin')['B_gaze'].mean()

                axes[1, col].plot(curve_A_gaze.index, curve_A_gaze.values, '-', color='#2E86AB', alpha=0.35, linewidth=1)
                axes[1, col].plot(curve_B_gaze.index, curve_B_gaze.values, '-', color='#C73E1D', alpha=0.35, linewidth=1)

        axes[0, col].set_title(f"Pull rate over training steps -- {phase}")
        axes[0, col].set_xlabel("Step (within phase)")
        axes[0, col].set_ylabel("P(pull | eligible)")
        axes[0, col].set_ylim(0, 1.05)

        axes[1, col].set_title(f"Gaze rate over training steps -- {phase}")
        axes[1, col].set_xlabel("Step (within phase)")
        axes[1, col].set_ylabel("P(gaze=1)")
        axes[1, col].set_ylim(0, 1.05)

    # --- Manual legend (since we plotted many thin unlabeled lines) ---
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2E86AB', lw=2, label='Agent A'),
                        Line2D([0], [0], color='#C73E1D', lw=2, label='Agent B')]
    for ax in axes.flat:
        ax.legend(handles=legend_elements, fontsize=8)

    plt.suptitle(f"Sanity check: does behavior change over training at all?\n"
                 f"({CHECKPOINT_SET}, every individual seed/order shown as a thin line, {len(SEEDS)*len(ORDERS)} runs/phase)", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_save_path+f"sanity_check_over_steps_{CHECKPOINT_SET}.png", dpi=150)
    print(f"Saved sanity_check_over_steps_{CHECKPOINT_SET}.png")


# In[ ]:


#################################################################
# COMPARE: pull rate, gaze rate, IPI, success rate vs force --
# across all 4 conditions (A/B varying x fresh/after-other-phase),
# combining orders A_then_B and B_then_A, seeds 0-5.
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if 0:

    CHECKPOINT_SET = "generalist_withGAE_PPO"
    ORDERS = ['A_then_B', 'B_then_A']
    SEEDS = (0, 1, 2, 3, 4, 5)
    LATE_FRAC = 0.2   # last 20% of each phase = "converged" window


    def get_varying_agent(order, phase):
        """Which agent varies force in this (order, phase) combination."""
        if order == 'A_then_B':
            return 'A' if phase == 'phase1' else 'B'
        else:  # B_then_A
            return 'B' if phase == 'phase1' else 'A'


    def condition_label(order, phase):
        agent = get_varying_agent(order, phase)
        is_fresh = (phase == 'phase1')
        other = 'B' if agent == 'A' else 'A'
        return f"{agent}_{'fresh' if is_fresh else f'after_{other}'}"


    def reconstruct_successes_labeled(log_df, sync_window=10):
        a_events = log_df.index[(log_df['A_pull'] == 1) & (log_df['can_pull_A'])].to_numpy()
        b_events = log_df.index[(log_df['B_pull'] == 1) & (log_df['can_pull_B'])].to_numpy()
        a_success = np.zeros(len(a_events), dtype=bool)
        if len(a_events) == 0 or len(b_events) == 0:
            return a_events, a_success
        used_b = set()
        for i, a_idx in enumerate(a_events):
            candidates = [c for c in b_events if abs(c - a_idx) <= sync_window and c not in used_b]
            if candidates:
                closest = min(candidates, key=lambda c: abs(c - a_idx))
                used_b.add(closest)
                a_success[i] = True
        return a_events, a_success


    # --- Collect per-seed binned curves, grouped by condition ---
    conditions = ['A_fresh', 'A_after_B', 'B_fresh', 'B_after_A']
    per_condition_pull = {c: [] for c in conditions}
    per_condition_gaze = {c: [] for c in conditions}
    per_condition_ipi = {c: [] for c in conditions}
    per_condition_success = {c: [] for c in conditions}
    per_condition_overall = {c: {'pull': [], 'gaze': [], 'ipi': [], 'success': []} for c in conditions}

    for order in ORDERS:
        for seed in SEEDS:
            fname = data_save_path+f"training_log_{order}_{CHECKPOINT_SET}_seed{seed}.pkl"
            try:
                log_df = pd.read_pickle(fname)
            except FileNotFoundError:
                print(f"Missing {fname}")
                continue

            for phase in ['phase1', 'phase2']:
                agent = get_varying_agent(order, phase)
                cond = condition_label(order, phase)

                phase_df = log_df[log_df['phase'] == phase].reset_index(drop=True)
                n = len(phase_df)
                late = phase_df.iloc[int(n * (1 - LATE_FRAC)):].copy()
                late['force_bin'] = (late[f'force_{agent}'] // 100) * 100

                # --- Pull rate vs force ---
                elig = late[late[f'can_pull_{agent}']]
                if len(elig) > 0:
                    per_condition_pull[cond].append(elig.groupby('force_bin')[f'{agent}_pull'].mean())
                    per_condition_overall[cond]['pull'].append(elig[f'{agent}_pull'].mean())

                # --- Gaze rate vs force (unconditioned) ---
                if f'{agent}_gaze' in late.columns:
                    per_condition_gaze[cond].append(late.groupby('force_bin')[f'{agent}_gaze'].mean())
                    per_condition_overall[cond]['gaze'].append(late[f'{agent}_gaze'].mean())

                # --- IPI vs force ---
                pull_col, can_pull_col = f'{agent}_pull', f'can_pull_{agent}'
                pull_idx = late.index[(late[pull_col] == 1) & (late[can_pull_col])].to_numpy()
                if len(pull_idx) >= 2:
                    ipi = np.diff(pull_idx)
                    force_at_start = late.loc[pull_idx[:-1], f'force_{agent}'].to_numpy()
                    ipi_df = pd.DataFrame({'force_bin': (force_at_start // 100) * 100, 'ipi': ipi})
                    per_condition_ipi[cond].append(ipi_df.groupby('force_bin')['ipi'].mean())
                    per_condition_overall[cond]['ipi'].append(ipi.mean())

                # --- Success rate vs force (only meaningful for the varying agent = A's role;
                #     success is a joint A/B event, so we track it relative to whichever
                #     agent is varying, using A's attempts as the anchor when agent=='A',
                #     or symmetric logic when agent=='B') ---
                if agent == 'A':
                    a_events, a_success = reconstruct_successes_labeled(late, sync_window=TARGET_WINDOW)
                    if len(a_events) > 0:
                        force_at_attempt = late.loc[a_events, 'force_A'].to_numpy()
                        succ_df = pd.DataFrame({'force_bin': (force_at_attempt // 100) * 100, 'success': a_success})
                        per_condition_success[cond].append(succ_df.groupby('force_bin')['success'].mean())
                        per_condition_overall[cond]['success'].append(a_success.mean() if len(a_success) > 0 else np.nan)
                else:  # agent == 'B' -- mirror the logic using B's attempts
                    b_events = late.index[(late['B_pull'] == 1) & (late['can_pull_B'])].to_numpy()
                    a_events_all = late.index[(late['A_pull'] == 1) & (late['can_pull_A'])].to_numpy()
                    b_success = np.zeros(len(b_events), dtype=bool)
                    used_a = set()
                    for i, b_idx in enumerate(b_events):
                        candidates = [c for c in a_events_all if abs(c - b_idx) <= TARGET_WINDOW and c not in used_a]
                        if candidates:
                            closest = min(candidates, key=lambda c: abs(c - b_idx))
                            used_a.add(closest)
                            b_success[i] = True
                    if len(b_events) > 0:
                        force_at_attempt = late.loc[b_events, 'force_B'].to_numpy()
                        succ_df = pd.DataFrame({'force_bin': (force_at_attempt // 100) * 100, 'success': b_success})
                        per_condition_success[cond].append(succ_df.groupby('force_bin')['success'].mean())
                        per_condition_overall[cond]['success'].append(b_success.mean() if len(b_success) > 0 else np.nan)


    # --- Print overall summary table ---
    print(f"{'Condition':>14} | {'Pull rate':>18} | {'Gaze rate':>18} | {'IPI':>16} | {'Success rate':>18}")
    for cond in conditions:
        pr = per_condition_overall[cond]['pull']
        gr = per_condition_overall[cond]['gaze']
        ip = per_condition_overall[cond]['ipi']
        sr = per_condition_overall[cond]['success']
        pr_s = f"{np.mean(pr):.4f} +/- {np.std(pr):.4f}" if pr else "NO DATA"
        gr_s = f"{np.mean(gr):.4f} +/- {np.std(gr):.4f}" if gr else "NO DATA"
        ip_s = f"{np.mean(ip):.2f} +/- {np.std(ip):.2f}" if ip else "NO DATA"
        sr_s = f"{np.nanmean(sr):.4f} +/- {np.nanstd(sr):.4f}" if sr else "NO DATA"
        print(f"{cond:>14} | {pr_s:>18} | {gr_s:>18} | {ip_s:>16} | {sr_s:>18}")


    # --- Plot: 2x2 grid, 4 conditions overlaid per panel ---
    colors = {'A_fresh': '#2E86AB', 'A_after_B': '#6FB1D6',
              'B_fresh': '#C73E1D', 'B_after_A': '#E8927C'}

    def plot_metric(ax, per_condition_dict, title, ylabel, ylim=None):
        for cond in conditions:
            if not per_condition_dict[cond]:
                continue
            combined = pd.concat(per_condition_dict[cond], axis=1)
            mean_curve, std_curve = combined.mean(axis=1), combined.std(axis=1)
            ax.plot(mean_curve.index, mean_curve.values, 'o-', label=cond, color=colors[cond])
            ax.fill_between(mean_curve.index, mean_curve.values - std_curve.values,
                             mean_curve.values + std_curve.values, alpha=0.15, color=colors[cond])
        ax.set_xlabel("Force (g)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        if ylim: ax.set_ylim(*ylim)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_metric(axes[0,0], per_condition_pull, "Attempt rate vs force", "P(pull | eligible)", ylim=(0,1.05))
    plot_metric(axes[0,1], per_condition_gaze, "Gaze rate vs force", "P(gaze=1)", ylim=(0,1.05))
    plot_metric(axes[1,0], per_condition_ipi, "Mean IPI vs force", "Steps")
    plot_metric(axes[1,1], per_condition_success, "Success rate vs force", "Success rate", ylim=(0,1.05))

    plt.suptitle(f"Two-phase training comparison ({CHECKPOINT_SET}, {len(SEEDS)} seeds, "
                 f"last {LATE_FRAC*100:.0f}% of each phase)", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_save_path+f"two_phase_comparison_{CHECKPOINT_SET}.png", dpi=150)
    print(f"\nSaved two_phase_comparison_{CHECKPOINT_SET}.png")


# ### do some sweeping testing using previously trained model. any of them can be tested 
# #### Here, i first load the previously trained model with save pth file, then test with newly generated testing dataset

# In[ ]:


#################################################################
# FLEXIBLE EVALUATION: sweep RELIABILITY x FORCE for ANY checkpoint
# (original gaze-only checkpoints OR effort-trained checkpoints),
# frozen weights, NO training -- generates a fresh test set for
# behavioral comparison across models.
#
# Paste into a new cell in your discreteEffort notebook (needs
# AgentBrain, CurriculumEnv, preprocess_state_with_effort,
# get_private_state, sample_actions, load_and_expand_agent, and the
# hyperparameter constants already defined earlier in the notebook).
#################################################################
import torch
import numpy as np
import pandas as pd
import os

def derive_eval_tag(agent_A_path, agent_B_path, fixed_agent):
    """
    Builds a short, descriptive identifier from the checkpoint paths and
    eval setup, so saved files reflect what was actually tested instead
    of a fixed generic name that silently overwrites the last run.
    e.g. "generalist_withGAE_PPO_fixedB" or
         "A_then_B_generalist_withGAE_PPO_phase2_seed0_fixedB"
    """
    def clean(path):
        name = os.path.basename(path).replace('.pth', '')
        for prefix in ('agent_A_', 'agent_B_'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    tag_A, tag_B = clean(agent_A_path), clean(agent_B_path)
    checkpoint_tag = tag_A if tag_A == tag_B else f"{tag_A}_vs_{tag_B}"
    return f"{checkpoint_tag}_fixed{fixed_agent}"


def load_agent_flexible(checkpoint_path, target_public_size=7, target_private_size=5,
                         old_public_size=5, old_private_size=5):
    """
    Auto-detects checkpoint input size and loads accordingly:
    - Already target-dim (e.g. 12): loads directly -- tests actual LEARNED
      force sensitivity, and reliability at potentially NOVEL values (since
      effort training held reliability fixed at 1.0 -- this is zero-shot).
    - Original old-dim (e.g. 10): expands via ZERO-INIT on the new force
      columns -- tests reliability sensitivity with the full original
      weights untouched; force sensitivity will be ~0 since those columns
      start at zero and were never trained (expected, not a bug).
    """
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    input_size = state['rnn_trunk.weight_ih'].shape[1]
    target_input_size = target_public_size + target_private_size

    model = AgentBrain(target_public_size, target_private_size, RNN_HIDDEN_SIZE,
                        MOVE_ACTIONS, PULL_ACTIONS, GAZE_ACTIONS)

    if input_size == target_input_size:
        model.load_state_dict(state)
        print(f"Loaded {checkpoint_path} directly ({input_size}-dim input, no expansion needed).")
    elif input_size == old_public_size + old_private_size:
        model = load_and_expand_agent(model, checkpoint_path,
                                       old_public_size=old_public_size,
                                       old_private_size=old_private_size,
                                       new_public_size=target_public_size,
                                       new_private_size=target_private_size)
        print(f"Loaded {checkpoint_path} via zero-init expansion "
              f"({input_size}-dim -> {target_input_size}-dim).")
    else:
        raise ValueError(f"Unexpected weight_ih input size {input_size} in {checkpoint_path} "
                          f"(expected {old_public_size+old_private_size} or {target_input_size})")
    model.eval()
    return model


def run_flexible_eval(agent_A_path, agent_B_path, reliability_levels, force_levels,
                       fixed_agent='B', steps_per_condition=3000, seed=0):
    """
    Frozen-weight evaluation across a RELIABILITY x FORCE grid. No training,
    no gradient updates -- pure behavioral measurement. Each (reliability,
    force) condition is run independently: env is reset, hidden states are
    reset, and the block-advance counter is reset, so conditions can't
    contaminate each other.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent_A = load_agent_flexible(agent_A_path)
    agent_B = load_agent_flexible(agent_B_path)

    env = CurriculumEnv(block_len=steps_per_condition + 10000, fixed_agent=fixed_agent)
    env.set_training_phase("COOP_FULL", sync_window=TARGET_WINDOW)
    env.gaze_cost = GAZE_COST

    log = []
    total_conditions = len(reliability_levels) * len(force_levels)
    done_count = 0

    for reliability in reliability_levels:
        for force in force_levels:
            if fixed_agent == 'A':
                env.force_A, env.force_B = FORCE_LOW, force
            else:
                env.force_A, env.force_B = force, FORCE_LOW
            env.cost_A = force_to_cost(env.force_A)
            env.cost_B = force_to_cost(env.force_B)
            env._effort_step_counter = 0  # prevent auto block-switch mid-condition

            state_A_dict, state_B_dict = env.reset()
            public_A = preprocess_state_with_effort(state_A_dict, env.force_A, env.force_B)
            public_B = preprocess_state_with_effort(state_B_dict, env.force_B, env.force_A)
            private_A = torch.zeros(1, PRIVATE_INPUT_SIZE)
            private_B = torch.zeros(1, PRIVATE_INPUT_SIZE)
            hidden_A = torch.zeros(1, RNN_HIDDEN_SIZE)
            hidden_B = torch.zeros(1, RNN_HIDDEN_SIZE)

            for t in range(steps_per_condition):
                mask_A = torch.tensor([[0., 0. if state_A_dict['can_pull'] else -1e9]]).float()
                mask_B = torch.tensor([[0., 0. if state_B_dict['can_pull'] else -1e9]]).float()

                with torch.no_grad():
                    logits_A, _, hidden_A = agent_A(public_A, private_A, hidden_A, mask_A)
                    logits_B, _, hidden_B = agent_B(public_B, private_B, hidden_B, mask_B)

                act_A, _ = sample_actions(logits_A)
                act_B, _ = sample_actions(logits_B)

                can_pull_A_now, can_pull_B_now = state_A_dict['can_pull'], state_B_dict['can_pull']
                next_A, next_B, r_A, r_B, done = env.step(act_A, act_B)

                log.append({'reliability': reliability, 'force_A': env.force_A, 'force_B': env.force_B,
                            'A_pull': act_A['pull'], 'B_pull': act_B['pull'],
                            'A_gaze': act_A['gaze'], 'B_gaze': act_B['gaze'],
                            'can_pull_A': can_pull_A_now, 'can_pull_B': can_pull_B_now})

                private_A = get_private_state(next_B, env.B_time_since_pull, env.current_sync_window, reliability, act_A['gaze']==1)
                private_B = get_private_state(next_A, env.A_time_since_pull, env.current_sync_window, reliability, act_B['gaze']==1)
                public_A = preprocess_state_with_effort(next_A, env.force_A, env.force_B)
                public_B = preprocess_state_with_effort(next_B, env.force_B, env.force_A)
                state_A_dict, state_B_dict = next_A, next_B

            done_count += 1
            if done_count % 10 == 0 or done_count == total_conditions:
                print(f"  ...{done_count}/{total_conditions} conditions done")

    return pd.DataFrame(log)


#################################################################
# USAGE -- change these two lines to test any checkpoint pair
#################################################################

# --- Example 1: the ORIGINAL gaze-only generalist (never saw force at all) ---
if 0:
    OLD_CHECKPOINT_DIR_TEST = "/gpfs/radev/home/ws523/marmoset_tracking_DLCv2/following_up_analysis"+    "/MARL_gaze_accumulation_strategies/training_withGAE_PPO_fixedPullCoolTime"
    #
    AGENT_A_PATH = f"{OLD_CHECKPOINT_DIR_TEST}/agent_A_generalist_withGAE_PPO.pth"
    AGENT_B_PATH = f"{OLD_CHECKPOINT_DIR_TEST}/agent_B_generalist_withGAE_PPO.pth"
#
# --- Example 2: an effort-trained checkpoint (uncomment to use instead) ---
if 1:
    # best one? B_then_A_generalist_withGAE_PPO_phase2_seed3
    
    AGENT_A_PATH = data_save_path+"agent_A_B_then_A_generalist_withGAE_PPO_phase2_seed3.pth"
    AGENT_B_PATH = data_save_path+"agent_B_B_then_A_generalist_withGAE_PPO_phase2_seed3.pth"
    

FIXED_AGENT = 'B'
# RELIABILITY_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
RELIABILITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# RELIABILITY_LEVELS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                       0.55, 0.6, 0.65, 0.7, 0.77, 0.8, 0.85, 0.9, 0.95, 1.0]
# FORCE_LEVELS = DISCRETE_FORCE_LEVELS.tolist()   # [100, 200, ..., 900]
FORCE_LEVELS = [100, 200, 300, 400, 500, 600, 700, 800, 900]
# FORCE_LEVELS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
STEPS_PER_CONDITION = 10000

EVAL_SEED = 3

FORCE_RERUN = 0   # set to 1 to force-redo even if the output file already exists

EVAL_TAG = derive_eval_tag(AGENT_A_PATH, AGENT_B_PATH, FIXED_AGENT)
OUT_NAME = data_save_path + f"flexible_eval_log_{EVAL_TAG}.pkl"

if not FORCE_RERUN and os.path.exists(OUT_NAME):
    print(f"Found existing {OUT_NAME} -- skipping eval, loading instead.")
    print("Set FORCE_RERUN = 1 to redo it.")
    eval_log = pd.read_pickle(OUT_NAME)
else:
    if FORCE_RERUN and os.path.exists(OUT_NAME):
        print(f"FORCE_RERUN=1 -- overwriting existing {OUT_NAME}\n")
    eval_log = run_flexible_eval(AGENT_A_PATH, AGENT_B_PATH, RELIABILITY_LEVELS, FORCE_LEVELS,
                                  fixed_agent=FIXED_AGENT, steps_per_condition=STEPS_PER_CONDITION, seed=EVAL_SEED)
    eval_log.to_pickle(OUT_NAME)
    print(f"\nSaved {OUT_NAME} ({len(eval_log)} steps across "
          f"{len(RELIABILITY_LEVELS)}x{len(FORCE_LEVELS)}={len(RELIABILITY_LEVELS)*len(FORCE_LEVELS)} conditions)")

print(f"Eval tag for this run: {EVAL_TAG}  <- use this to load the matching file in the plotting cell")


# In[ ]:


#################################################################
# PLOT: pull rate, gaze rate, IPI vs reliability and force, ONE
# POINT PER CONDITION (no error bars -- each point is that
# condition's own mean), with a regression line + p-value fit
# across CONDITIONS (not pooled raw trial data).
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

EVAL_TAG = derive_eval_tag(AGENT_A_PATH, AGENT_B_PATH, FIXED_AGENT)
eval_log = pd.read_pickle(data_save_path + f"flexible_eval_log_{EVAL_TAG}.pkl")

print(f"Loaded {len(eval_log)} rows.")

force_A_varies = eval_log['force_A'].nunique() > 1
varying_force_col = 'force_A' if force_A_varies else 'force_B'
varying_agent_pull = 'A_pull' if force_A_varies else 'B_pull'
print(f"Varying force column detected: {varying_force_col}\n")


def get_ipi_per_condition(df, agent):
    """
    Returns a DataFrame with one row per (reliability, force) condition:
    mean IPI within that condition. Computed within each condition
    separately so diff() never crosses a condition boundary.
    """
    rows = []
    for (rel, force), group in df.groupby(['reliability', varying_force_col]):
        group = group.reset_index(drop=True)
        pull_col, can_pull_col = f'{agent}_pull', f'can_pull_{agent}'
        pull_idx = group.index[(group[pull_col] == 1) & (group[can_pull_col])].to_numpy()
        if len(pull_idx) < 2:
            continue
        ipi = np.diff(pull_idx)
        rows.append({'reliability': rel, varying_force_col: force, 'ipi_mean': ipi.mean()})
    return pd.DataFrame(rows)


def condition_level_stats(df, agent, metric, x_col):
    """
    Returns (x_values, y_values, slope, pvalue, n_conditions) for one
    metric ('pull', 'gaze', 'ipi'), with ONE VALUE PER CONDITION
    (reliability x force cell) -- no marginalization/pooling across the
    other axis, no error bars. Regression is fit across these
    condition-level points (n = number of conditions), not raw trials.
    """
    if metric == 'pull':
        elig = df[df[f'can_pull_{agent}']]
        cond_df = elig.groupby(['reliability', varying_force_col])[f'{agent}_pull'].mean().reset_index()
        y_col = f'{agent}_pull'

    elif metric == 'gaze':
        cond_df = df.groupby(['reliability', varying_force_col])[f'{agent}_gaze'].mean().reset_index()
        y_col = f'{agent}_gaze'

    elif metric == 'ipi':
        cond_df = get_ipi_per_condition(df, agent)
        if len(cond_df) < 4:
            return None
        y_col = 'ipi_mean'

    else:
        raise ValueError(f"unknown metric: {metric}")

    x_vals = cond_df[x_col].values.astype(float)
    y_vals = cond_df[y_col].values.astype(float)
    n_conditions = len(cond_df)

    if n_conditions < 3 or np.std(x_vals) == 0:
        slope, pval = np.nan, np.nan
    else:
        slope, intercept, r, pval, se = stats.linregress(x_vals, y_vals)

    return x_vals, y_vals, slope, pval, n_conditions


def plot_with_regression(ax, df, agent, metric, x_col, ylabel, ylim=None, color='#2E86AB'):
    result = condition_level_stats(df, agent, metric, x_col)
    if result is None:
        ax.set_title(f"Agent {agent}: {metric} vs {x_col}\n(not enough data)")
        return

    x_vals, y_vals, slope, pval, n_conditions = result

    ax.scatter(x_vals, y_vals, color=color, s=25, alpha=0.8, label='per-condition mean')

    if not np.isnan(slope):
        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
        intercept = np.mean(y_vals) - slope * np.mean(x_vals)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='black', linewidth=1.5, alpha=0.7)
        sig_marker = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        title_stats = f"slope={slope:.5f}, p={pval:.4f} ({sig_marker}), n_conditions={n_conditions}"
    else:
        title_stats = f"n_conditions={n_conditions} (insufficient variance for regression)"

    ax.set_xlabel(x_col if x_col != varying_force_col else f"{x_col} (g)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Agent {agent}: {ylabel} vs {x_col}\n{title_stats}")
    if ylim: ax.set_ylim(*ylim)


# --- Figure 1: vs reliability ---
fig1, axes1 = plt.subplots(2, 3, figsize=(19, 10))
plot_with_regression(axes1[0,0], eval_log, 'A', 'pull', 'reliability', "P(pull|eligible)", (0,1.05))
plot_with_regression(axes1[0,1], eval_log, 'A', 'gaze', 'reliability', "P(gaze=1)", (0,1.05))
plot_with_regression(axes1[0,2], eval_log, 'A', 'ipi', 'reliability', "Mean IPI (steps)")
plot_with_regression(axes1[1,0], eval_log, 'B', 'pull', 'reliability', "P(pull|eligible)", (0,1.05), color='#C73E1D')
plot_with_regression(axes1[1,1], eval_log, 'B', 'gaze', 'reliability', "P(gaze=1)", (0,1.05), color='#C73E1D')
plot_with_regression(axes1[1,2], eval_log, 'B', 'ipi', 'reliability', "Mean IPI (steps)", color='#C73E1D')
plt.suptitle("Behavior vs RELIABILITY, one point per condition, with regression + p-value", y=1.02)
plt.tight_layout()
plt.savefig(fig_save_path + f"flexible_eval_vs_reliability_stats_{EVAL_TAG}.pdf", dpi=150)
print("Saved flexible_eval_vs_reliability_stats.pdf")

# --- Figure 2: vs force ---
fig2, axes2 = plt.subplots(2, 3, figsize=(19, 10))
plot_with_regression(axes2[0,0], eval_log, 'A', 'pull', varying_force_col, "P(pull|eligible)", (0,1.05))
plot_with_regression(axes2[0,1], eval_log, 'A', 'gaze', varying_force_col, "P(gaze=1)", (0,1.05))
plot_with_regression(axes2[0,2], eval_log, 'A', 'ipi', varying_force_col, "Mean IPI (steps)")
plot_with_regression(axes2[1,0], eval_log, 'B', 'pull', varying_force_col, "P(pull|eligible)", (0,1.05), color='#C73E1D')
plot_with_regression(axes2[1,1], eval_log, 'B', 'gaze', varying_force_col, "P(gaze=1)", (0,1.05), color='#C73E1D')
plot_with_regression(axes2[1,2], eval_log, 'B', 'ipi', varying_force_col, "Mean IPI (steps)", color='#C73E1D')
plt.suptitle(f"Behavior vs FORCE ({varying_force_col}), one point per condition, with regression + p-value", y=1.02)
plt.tight_layout()
plt.savefig(fig_save_path + f"flexible_eval_vs_force_stats_{EVAL_TAG}.pdf", dpi=150)
print("Saved flexible_eval_vs_force_stats.pdf")

print("\nEach point is one (reliability, force) condition's own mean -- no marginalization")
print("across the other axis, no error bars. Regression is fit across these")
print("condition-level points (n_conditions), not pooled raw trial data.")


# In[ ]:


#################################################################
# NEURAL-LOGGING EVAL: same reliability x force grid design as
# run_flexible_eval, but ALSO captures the continuous hidden-state
# trace (the "neurons") for both agents, saved as separate .npy
# arrays aligned row-by-row with the behavioral log.
#
# Scoped to a SMALL subset grid first (3 reliability x 4 force) to
# validate the pipeline cheaply before committing to the full grid.
#################################################################
import torch
import numpy as np
import pandas as pd
import os

def run_flexible_eval_with_neural(agent_A_path, agent_B_path, reliability_levels, force_levels,
                                    fixed_agent='B', steps_per_condition=3000, seed=0):
    """
    Same structure as run_flexible_eval, but also returns continuous
    hidden-state traces for both agents (shape: [n_total_steps, hidden_size]),
    plus 'condition_id' and 'step_within_condition' columns in the log so
    PETH extraction can safely avoid windows that would cross a condition
    boundary (conditions are otherwise unrelated -- different reliability
    and/or force).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent_A = load_agent_flexible(agent_A_path)
    agent_B = load_agent_flexible(agent_B_path)

    env = CurriculumEnv(block_len=steps_per_condition + 10000, fixed_agent=fixed_agent)
    env.set_training_phase("COOP_FULL", sync_window=TARGET_WINDOW)
    env.gaze_cost = GAZE_COST

    log = []
    hidden_A_trace = []
    hidden_B_trace = []

    total_conditions = len(reliability_levels) * len(force_levels)
    done_count = 0
    condition_id = 0

    for reliability in reliability_levels:
        for force in force_levels:
            if fixed_agent == 'A':
                env.force_A, env.force_B = FORCE_LOW, force
            else:
                env.force_A, env.force_B = force, FORCE_LOW
            env.cost_A = force_to_cost(env.force_A)
            env.cost_B = force_to_cost(env.force_B)
            env._effort_step_counter = 0

            state_A_dict, state_B_dict = env.reset()
            public_A = preprocess_state_with_effort(state_A_dict, env.force_A, env.force_B)
            public_B = preprocess_state_with_effort(state_B_dict, env.force_B, env.force_A)
            private_A = torch.zeros(1, PRIVATE_INPUT_SIZE)
            private_B = torch.zeros(1, PRIVATE_INPUT_SIZE)
            hidden_A = torch.zeros(1, RNN_HIDDEN_SIZE)
            hidden_B = torch.zeros(1, RNN_HIDDEN_SIZE)

            for t in range(steps_per_condition):
                mask_A = torch.tensor([[0., 0. if state_A_dict['can_pull'] else -1e9]]).float()
                mask_B = torch.tensor([[0., 0. if state_B_dict['can_pull'] else -1e9]]).float()

                with torch.no_grad():
                    logits_A, _, hidden_A = agent_A(public_A, private_A, hidden_A, mask_A)
                    logits_B, _, hidden_B = agent_B(public_B, private_B, hidden_B, mask_B)

                # Log hidden state AFTER this step's update -- i.e. the state
                # that reflects having just observed the current input and
                # produced the current action.
                hidden_A_trace.append(hidden_A.numpy().copy().flatten())
                hidden_B_trace.append(hidden_B.numpy().copy().flatten())

                act_A, _ = sample_actions(logits_A)
                act_B, _ = sample_actions(logits_B)

                can_pull_A_now, can_pull_B_now = state_A_dict['can_pull'], state_B_dict['can_pull']
                next_A, next_B, r_A, r_B, done = env.step(act_A, act_B)

                log.append({'condition_id': condition_id, 'step_within_condition': t,
                            'reliability': reliability, 'force_A': env.force_A, 'force_B': env.force_B,
                            'A_pull': act_A['pull'], 'B_pull': act_B['pull'],
                            'A_gaze': act_A['gaze'], 'B_gaze': act_B['gaze'],
                            'can_pull_A': can_pull_A_now, 'can_pull_B': can_pull_B_now})

                private_A = get_private_state(next_B, env.B_time_since_pull, env.current_sync_window, reliability, act_A['gaze']==1)
                private_B = get_private_state(next_A, env.A_time_since_pull, env.current_sync_window, reliability, act_B['gaze']==1)
                public_A = preprocess_state_with_effort(next_A, env.force_A, env.force_B)
                public_B = preprocess_state_with_effort(next_B, env.force_B, env.force_A)
                state_A_dict, state_B_dict = next_A, next_B

            condition_id += 1
            done_count += 1
            if done_count % 5 == 0 or done_count == total_conditions:
                print(f"  ...{done_count}/{total_conditions} conditions done")

    log_df = pd.DataFrame(log)
    hidden_A_array = np.array(hidden_A_trace, dtype=np.float32)
    hidden_B_array = np.array(hidden_B_trace, dtype=np.float32)
    return log_df, hidden_A_array, hidden_B_array


def run_neural_eval_with_cache(agent_A_path, agent_B_path, reliability_levels, force_levels,
                                 fixed_agent, steps_per_condition, seed,
                                 data_save_path, eval_tag, force_rerun=False):
    """Same skip-if-exists / force_rerun pattern used elsewhere in the notebook."""
    log_path = data_save_path + f"neural_log_{eval_tag}.pkl"
    hA_path = data_save_path + f"neural_hiddenA_{eval_tag}.npy"
    hB_path = data_save_path + f"neural_hiddenB_{eval_tag}.npy"

    if not force_rerun and all(os.path.exists(p) for p in (log_path, hA_path, hB_path)):
        print(f"Found existing neural data for tag={eval_tag} -- loading instead of re-running.")
        return pd.read_pickle(log_path), np.load(hA_path), np.load(hB_path)

    if force_rerun and os.path.exists(log_path):
        print(f"force_rerun=True -- overwriting existing neural data for tag={eval_tag}\n")

    log_df, hidden_A_array, hidden_B_array = run_flexible_eval_with_neural(
        agent_A_path, agent_B_path, reliability_levels, force_levels,
        fixed_agent=fixed_agent, steps_per_condition=steps_per_condition, seed=seed)

    log_df.to_pickle(log_path)
    np.save(hA_path, hidden_A_array)
    np.save(hB_path, hidden_B_array)
    print(f"Saved {log_path}")
    print(f"Saved {hA_path} (shape {hidden_A_array.shape})")
    print(f"Saved {hB_path} (shape {hidden_B_array.shape})")
    return log_df, hidden_A_array, hidden_B_array


#################################################################
# PETH EXTRACTION: pull-aligned hidden-state windows, safely
# excluding events too close to a condition boundary (VERIFIED via
# sandbox test: extracted windows never cross into a neighboring
# condition's data).
#################################################################

def extract_peth_windows(log_df, hidden_array, agent, window_steps=40):
    """
    Returns:
      peth_array: shape (n_events, 2*window_steps+1, hidden_size)
      event_meta: DataFrame with one row per event (reliability, force, condition_id)
    Excludes any pull event whose window would cross a condition boundary
    (i.e. events within `window_steps` of the start or end of their own
    condition's step range).
    """
    pull_col, can_pull_col = f'{agent}_pull', f'can_pull_{agent}'
    steps_per_condition = log_df['step_within_condition'].max() + 1

    eligible_events = log_df[
        (log_df[pull_col] == 1) & (log_df[can_pull_col]) &
        (log_df['step_within_condition'] >= window_steps) &
        (log_df['step_within_condition'] <= steps_per_condition - 1 - window_steps)
    ]

    peth_list = []
    meta_list = []
    for idx, row in eligible_events.iterrows():
        window = hidden_array[idx - window_steps: idx + window_steps + 1]
        if window.shape[0] != 2 * window_steps + 1:
            continue  # safety net, shouldn't trigger given the filtering above
        peth_list.append(window)
        meta_list.append({'reliability': row['reliability'], 'force_A': row['force_A'],
                           'force_B': row['force_B'], 'condition_id': row['condition_id']})

    peth_array = np.stack(peth_list, axis=0) if peth_list else np.empty((0, 2*window_steps+1, hidden_array.shape[1]))
    event_meta = pd.DataFrame(meta_list)
    print(f"Agent {agent}: extracted {len(peth_list)} pull-aligned windows "
          f"(excluded events within {window_steps} steps of a condition boundary).")
    return peth_array, event_meta


#################################################################
# USAGE
#################################################################

# Subset grid to validate the pipeline first (full 10x17 grid later,
# once this is confirmed working)
NEURAL_RELIABILITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NEURAL_FORCE_LEVELS = [100, 200, 300, 400, 500, 600, 700, 800, 900]
NEURAL_STEPS_PER_CONDITION = 3000
NEURAL_SEED = 0
PETH_WINDOW_STEPS = 40   # +/- 4s, given TARGET_WINDOW=10 steps/second

DO_force_rerun = 0

NEURAL_TAG = derive_eval_tag(AGENT_A_PATH, AGENT_B_PATH, FIXED_AGENT) + "_neuralsubset"

neural_log, hidden_A, hidden_B = run_neural_eval_with_cache(
    AGENT_A_PATH, AGENT_B_PATH, NEURAL_RELIABILITY_LEVELS, NEURAL_FORCE_LEVELS,
    fixed_agent=FIXED_AGENT, steps_per_condition=NEURAL_STEPS_PER_CONDITION, seed=NEURAL_SEED,
    data_save_path=data_save_path, eval_tag=NEURAL_TAG, force_rerun=DO_force_rerun)

peth_A, meta_A = extract_peth_windows(neural_log, hidden_A, 'A', window_steps=PETH_WINDOW_STEPS)
peth_B, meta_B = extract_peth_windows(neural_log, hidden_B, 'B', window_steps=PETH_WINDOW_STEPS)


# In[ ]:


#################################################################
# PETH OVERLAY PLOTS: instead of small multiples (hard to compare
# across separate panels), overlays multiple conditions on shared
# axes -- one set of panels overlaying FORCE within each reliability
# level, another overlaying RELIABILITY within each force level.
# Saved as PDF.
#################################################################
import numpy as np
import matplotlib.pyplot as plt


def aggregate_peth_by_condition(peth_array, event_meta, varying_agent_force_col):
    """
    peth_array: (n_events, window_len, hidden_size) from extract_peth_windows
    event_meta: matching DataFrame with 'reliability' and force columns
    Returns dict {(reliability, force): {'mean_trace','sem_trace','n_events'}}
    """
    results = {}
    pop_trace_per_event = peth_array.mean(axis=2)  # mean across units -> (n_events, window_len)
    event_meta = event_meta.reset_index(drop=True)
    for (rel, force), group in event_meta.groupby(['reliability', varying_agent_force_col]):
        idx = group.index.to_numpy()
        traces = pop_trace_per_event[idx]
        results[(rel, force)] = {
            'mean_trace': traces.mean(axis=0),
            'sem_trace': traces.std(axis=0) / np.sqrt(len(traces)),
            'n_events': len(traces),
        }
    return results


def plot_peth_overlay_by_force(peth_results, agent, reliability_levels, force_levels, window_steps=PETH_WINDOW_STEPS):
    """One panel per reliability level; within each panel, all force levels overlaid as colored lines."""
    x_axis = np.arange(-window_steps, window_steps + 1) / 10.0
    n_rel = len(reliability_levels)
    fig, axes = plt.subplots(1, n_rel, figsize=(4.5*n_rel, 4), sharey=True)
    if n_rel == 1: axes = [axes]

    cmap = plt.cm.viridis(np.linspace(0, 1, len(force_levels)))

    for i, rel in enumerate(reliability_levels):
        ax = axes[i]
        for j, force in enumerate(force_levels):
            key = (rel, force)
            if key in peth_results:
                r = peth_results[key]
                ax.plot(x_axis, r['mean_trace'], color=cmap[j], label=f"{force}g (n={r['n_events']})")
                ax.fill_between(x_axis, r['mean_trace']-r['sem_trace'], r['mean_trace']+r['sem_trace'],
                                 alpha=0.15, color=cmap[j])
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"Reliability = {rel}")
        ax.set_xlabel("Time from pull (s)")
        if i == 0:
            ax.set_ylabel("Mean population activation")
        ax.legend(fontsize=7, title="Force")

    plt.suptitle(f"Agent {agent}: pull-aligned activity, force levels overlaid within each reliability panel", y=1.02)
    plt.tight_layout()
    return fig


def plot_peth_overlay_by_reliability(peth_results, agent, reliability_levels, force_levels, window_steps=PETH_WINDOW_STEPS):
    """One panel per force level; within each panel, all reliability levels overlaid as colored lines."""
    x_axis = np.arange(-window_steps, window_steps + 1) / 10.0
    n_force = len(force_levels)
    fig, axes = plt.subplots(1, n_force, figsize=(4.5*n_force, 4), sharey=True)
    if n_force == 1: axes = [axes]

    cmap = plt.cm.plasma(np.linspace(0, 1, len(reliability_levels)))

    for j, force in enumerate(force_levels):
        ax = axes[j]
        for i, rel in enumerate(reliability_levels):
            key = (rel, force)
            if key in peth_results:
                r = peth_results[key]
                ax.plot(x_axis, r['mean_trace'], color=cmap[i], label=f"rel={rel} (n={r['n_events']})")
                ax.fill_between(x_axis, r['mean_trace']-r['sem_trace'], r['mean_trace']+r['sem_trace'],
                                 alpha=0.15, color=cmap[i])
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"Force = {force}g")
        ax.set_xlabel("Time from pull (s)")
        if j == 0:
            ax.set_ylabel("Mean population activation")
        ax.legend(fontsize=7, title="Reliability")

    plt.suptitle(f"Agent {agent}: pull-aligned activity, reliability levels overlaid within each force panel", y=1.02)
    plt.tight_layout()
    return fig


#################################################################
# USAGE -- uses your existing variable names directly
#################################################################
varying_force_col = 'force_A' if FIXED_AGENT == 'B' else 'force_B'

peth_results_A = aggregate_peth_by_condition(peth_A, meta_A, varying_force_col)
peth_results_B = aggregate_peth_by_condition(peth_B, meta_B, varying_force_col)

for agent, results in [('A', peth_results_A), ('B', peth_results_B)]:
    fig1 = plot_peth_overlay_by_force(results, agent, NEURAL_RELIABILITY_LEVELS, NEURAL_FORCE_LEVELS)
    fig1.savefig(fig_save_path + f"peth_overlay_by_force_agent{agent}_{NEURAL_TAG}.pdf", format='pdf')
    print(f"Saved peth_overlay_by_force_agent{agent}_{NEURAL_TAG}.pdf")

    fig2 = plot_peth_overlay_by_reliability(results, agent, NEURAL_RELIABILITY_LEVELS, NEURAL_FORCE_LEVELS)
    fig2.savefig(fig_save_path + f"peth_overlay_by_reliability_agent{agent}_{NEURAL_TAG}.pdf", format='pdf')
    print(f"Saved peth_overlay_by_reliability_agent{agent}_{NEURAL_TAG}.pdf")


# In[ ]:


#################################################################
# DIAGNOSTIC: is the dip at t=0 in the population-mean PETH a real,
# widespread effect across units, or an artifact of a small subset
# of units with large swings dominating the average? Uses the raw
# per-unit peth_A/peth_B arrays you already have -- no rerun needed.
#################################################################
import numpy as np
import matplotlib.pyplot as plt


def diagnose_dip_at_zero(peth_array, window_steps=PETH_WINDOW_STEPS, baseline_window=(0, 10)):
    """
    peth_array: (n_events, window_len, hidden_size)
    For each unit, computes its own mean-across-events trace, then compares
    its value AT t=0 to its own pre-pull baseline (steps baseline_window,
    i.e. roughly -4s to -3s). Returns per-unit deltas so we see the
    DISTRIBUTION the population mean is summarizing, not just one number.
    """
    unit_mean_traces = peth_array.mean(axis=0)  # (window_len, hidden_size)
    t_zero_idx = window_steps

    baseline = unit_mean_traces[baseline_window[0]:baseline_window[1], :].mean(axis=0)
    at_zero = unit_mean_traces[t_zero_idx, :]
    delta = at_zero - baseline   # negative = this unit dipped at t=0

    n_dip = (delta < 0).sum()
    n_rise = (delta > 0).sum()
    return delta, unit_mean_traces, n_dip, n_rise


def plot_unit_heterogeneity(unit_mean_traces, delta, agent, window_steps=PETH_WINDOW_STEPS):
    """
    Panel 1: all individual unit traces overlaid (transparent gray), with
    the population mean in bold red on top -- shows directly whether units
    move together (real effect) or cancel out (averaging artifact).
    Panel 2: histogram of per-unit delta at t=0 -- the distribution behind
    the single population-mean number.
    """
    x_axis = np.arange(-window_steps, window_steps + 1) / 10.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for u in range(unit_mean_traces.shape[1]):
        axes[0].plot(x_axis, unit_mean_traces[:, u], color='gray', alpha=0.15, linewidth=0.6)
    axes[0].plot(x_axis, unit_mean_traces.mean(axis=1), color='#C73E1D', linewidth=2.5, label='Population mean')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel("Time from pull (s)")
    axes[0].set_ylabel("Unit activation")
    axes[0].set_title(f"Agent {agent}: all {unit_mean_traces.shape[1]} units (gray) + population mean (red)")
    axes[0].legend()

    axes[1].hist(delta, bins=30, color='#2E86AB', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].axvline(delta.mean(), color='#C73E1D', linewidth=2, label=f'mean={delta.mean():.4f}')
    axes[1].set_xlabel("Per-unit delta at t=0 (value - pre-pull baseline)")
    axes[1].set_ylabel("Number of units")
    axes[1].set_title("Distribution of per-unit effects")
    axes[1].legend()

    plt.tight_layout()
    return fig


#################################################################
# USAGE -- run for both agents, pooling across all conditions
# (peth_A/peth_B already contain events from every reliability x
# force condition combined)
#################################################################
for agent, peth_array in [('A', peth_A), ('B', peth_B)]:
    delta, unit_traces, n_dip, n_rise = diagnose_dip_at_zero(peth_array)

    print(f"\n=== Agent {agent} ===")
    print(f"{n_dip}/{len(delta)} units DIP at t=0, {n_rise}/{len(delta)} units RISE at t=0")
    print(f"Mean delta: {delta.mean():.5f}, Median delta: {np.median(delta):.5f}")
    print(f"Std across units: {delta.std():.5f}, Range: [{delta.min():.5f}, {delta.max():.5f}]")

    sorted_abs_delta = np.sort(np.abs(delta))[::-1]
    top10_share = sorted_abs_delta[:10].sum() / np.abs(delta).sum()
    even_share = 10 / len(delta)
    print(f"Top 10 units account for {top10_share*100:.1f}% of total |delta| "
          f"(even distribution would be {even_share*100:.1f}%)")
    if top10_share > 3 * even_share:
        print("  -> A SMALL SUBSET of units is dominating the population mean.")
    else:
        print("  -> Effect looks reasonably WIDESPREAD across units.")

    fig = plot_unit_heterogeneity(unit_traces, delta, agent)
    fig.savefig(fig_save_path + f"unit_heterogeneity_agent{agent}_{NEURAL_TAG}.pdf", format='pdf')
    print(f"Saved unit_heterogeneity_agent{agent}_{NEURAL_TAG}.pdf")

print("\n--- How to read this ---")
print("If ~half the units dip and ~half rise (like a coin flip), and no small")
print("subset dominates: the population-mean dip is a WEAK/borderline summary,")
print("not strong evidence of coordinated suppression -- individual units are")
print("doing different things that happen to net out slightly negative.")
print("If the large majority of units genuinely dip, and the effect isn't just")
print("a few outliers: that's real evidence of a shared, coordinated signal at")
print("the moment of the pull decision.")


# In[ ]:


#################################################################
# SIMPLE PER-VARIABLE UNIT TUNING: does each RNN unit's activity
# relate to self_pull, self_gaze, partner_pull, partner_gaze,
# reliability, self_force, partner_force? No GLM, no p-values --
# effect size only (Cohen's d for binary variables, Pearson r at
# the condition level for continuous ones, using only the PRE-PULL
# window of each pull event).
#
# VERIFIED via sandbox testing on a fake/untrained checkpoint:
# self_gaze/partner_gaze correctly show ~0 tuned units (no
# mechanical confound there); self_pull/partner_pull required
# restricting the comparison to ELIGIBLE steps only (can_pull==True)
# to remove a real confound -- eligibility is BOTH a direct network
# input AND the thing that mechanically gates whether a pull is even
# possible, so comparing "pulled" vs "couldn't possibly pull" was
# detecting that mechanical gating, not genuine tuning.
#################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EFFECT_THRESHOLD_BINARY = 0.8     # Cohen's d threshold for self_pull/self_gaze/partner_pull/partner_gaze
EFFECT_THRESHOLD_CONTINUOUS = 0.1  # Pearson |r| threshold for reliability/force -- higher than the binary
                                    # threshold because condition-level n is small (~12), where |r|>=0.3 is
                                    # only about 1 SE under pure noise and too lenient to trust alone.


def compute_binary_tuning(log_df, hidden_arr, agent, other):
    """
    Cohen's d per unit for each binary behavioral variable. Pull comparisons
    are restricted to ELIGIBLE steps only (can_pull==True) -- comparing
    "attempted" vs "chose not to, though able to" -- since comparing against
    steps where a pull wasn't even possible confounds tuning with the
    environment's mechanical eligibility gating. Gaze isn't gated this way,
    so no restriction needed there.
    """
    can_pull_self = log_df[f'can_pull_{agent}'].values.astype(bool)
    can_pull_other = log_df[f'can_pull_{other}'].values.astype(bool)

    variables = {
        'self_pull': (log_df[f'{agent}_pull'].values.astype(bool), can_pull_self),
        'self_gaze': (log_df[f'{agent}_gaze'].values.astype(bool), np.ones(len(log_df), dtype=bool)),
        'partner_pull': (log_df[f'{other}_pull'].values.astype(bool), can_pull_other),
        'partner_gaze': (log_df[f'{other}_gaze'].values.astype(bool), np.ones(len(log_df), dtype=bool)),
    }
    results = {}
    for var_name, (var_vals, eligible_mask) in variables.items():
        pos = hidden_arr[eligible_mask & var_vals]
        neg = hidden_arr[eligible_mask & ~var_vals]
        if len(pos) < 2 or len(neg) < 2:
            results[var_name] = np.full(hidden_arr.shape[1], np.nan)
            continue
        mean1, mean0 = pos.mean(axis=0), neg.mean(axis=0)
        std1, std0 = pos.std(axis=0, ddof=1), neg.std(axis=0, ddof=1)
        n1, n0 = len(pos), len(neg)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n0 - 1) * std0**2) / (n1 + n0 - 2))
        results[var_name] = (mean1 - mean0) / np.where(pooled_std > 0, pooled_std, np.nan)
    return pd.DataFrame(results)


def compute_condition_level_tuning(peth_array, event_meta, agent, window_steps=None):
    """
    Pearson r between each unit's per-CONDITION mean activity and
    reliability / self_force / partner_force -- computed at the condition
    level (not per-timestep), since these variables are constant within a
    condition and a per-timestep correlation would massively overstate the
    real sample size (pseudoreplication).

    Uses only the PRE-PULL window (-4s to 0s) of each pull event's PETH
    snippet (peth_array, event_meta -- from your extract_peth_windows),
    averaged first across events within a condition, then correlated
    against reliability/force across conditions. Restricting to the
    pre-pull window focuses on the decision-relevant period rather than
    diluting with drinking/idle time elsewhere in the condition.
    window_steps: defaults to PETH_WINDOW_STEPS if not given (must match
    whatever window extract_peth_windows used).
    """
    if window_steps is None:
        window_steps = PETH_WINDOW_STEPS

    pre_pull = peth_array[:, :window_steps + 1, :]           # -4s to 0s inclusive
    pre_pull_mean_per_event = pre_pull.mean(axis=1)            # (n_events, n_units)

    event_meta = event_meta.reset_index(drop=True)
    cond_ids = sorted(event_meta['condition_id'].unique())
    n_units = pre_pull_mean_per_event.shape[1]

    mean_activity_per_cond = np.zeros((len(cond_ids), n_units))
    reliability_per_cond = np.zeros(len(cond_ids))
    force_A_per_cond = np.zeros(len(cond_ids))
    force_B_per_cond = np.zeros(len(cond_ids))

    for i, cid in enumerate(cond_ids):
        mask = event_meta['condition_id'].values == cid
        mean_activity_per_cond[i] = pre_pull_mean_per_event[mask].mean(axis=0)
        reliability_per_cond[i] = event_meta.loc[mask, 'reliability'].iloc[0]
        force_A_per_cond[i] = event_meta.loc[mask, 'force_A'].iloc[0]
        force_B_per_cond[i] = event_meta.loc[mask, 'force_B'].iloc[0]

    self_force_per_cond = force_A_per_cond if agent == 'A' else force_B_per_cond
    partner_force_per_cond = force_B_per_cond if agent == 'A' else force_A_per_cond

    results = {}
    for var_name, var_vals in [('reliability', reliability_per_cond),
                                 ('self_force', self_force_per_cond),
                                 ('partner_force', partner_force_per_cond)]:
        if np.std(var_vals) == 0:
            results[var_name] = np.full(n_units, np.nan)
            print(f"  ({var_name} is constant in this eval for agent {agent} -- can't correlate, all NaN)")
            continue
        r_vals = np.array([np.corrcoef(mean_activity_per_cond[:, u], var_vals)[0, 1]
                            for u in range(n_units)])
        results[var_name] = r_vals
    return pd.DataFrame(results), len(cond_ids)


def run_simple_tuning(log_df, hidden_arr, peth_array, event_meta, agent):
    other = 'B' if agent == 'A' else 'A'
    binary_df = compute_binary_tuning(log_df, hidden_arr, agent, other)
    cond_df, n_conditions = compute_condition_level_tuning(peth_array, event_meta, agent)
    combined = pd.concat([binary_df, cond_df], axis=1)
    combined.insert(0, 'unit', np.arange(hidden_arr.shape[1]))
    print(f"Agent {agent}: condition-level correlations (pre-pull window) based on n={n_conditions} conditions "
          f"(small sample -- using threshold {EFFECT_THRESHOLD_CONTINUOUS}, not {EFFECT_THRESHOLD_BINARY})")
    return combined


def plot_tuning_heatmap(tuning_df, agent, fig_save_path, tag):
    cols = ['self_pull', 'self_gaze', 'partner_pull', 'partner_gaze', 'reliability', 'self_force', 'partner_force']
    data = tuning_df[cols].values
    order = np.argsort(-np.nanmax(np.abs(data), axis=1))
    data_sorted = data[order]

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(data_sorted, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_ylabel(f"Unit (sorted by max |effect|, n={data.shape[0]})")
    ax.set_title(f"Agent {agent}: per-unit tuning\n(Cohen's d for binary vars, Pearson r for reliability/force, pre-pull window)")
    plt.colorbar(im, ax=ax, label="Effect size")
    plt.tight_layout()
    fig.savefig(fig_save_path + f"unit_tuning_heatmap_agent{agent}_{tag}.pdf", format='pdf')
    print(f"Saved unit_tuning_heatmap_agent{agent}_{tag}.pdf")
    return fig


#################################################################
# USAGE -- uses your existing peth_A/meta_A/peth_B/meta_B
#################################################################
tuning_A = run_simple_tuning(neural_log, hidden_A, peth_A, meta_A, 'A')
tuning_B = run_simple_tuning(neural_log, hidden_B, peth_B, meta_B, 'B')

for agent, tuning_df in [('A', tuning_A), ('B', tuning_B)]:
    print(f"\n--- Agent {agent}: units exceeding threshold per variable ---")
    for col in ['self_pull', 'self_gaze', 'partner_pull', 'partner_gaze']:
        n = (tuning_df[col].abs() >= EFFECT_THRESHOLD_BINARY).sum()
        print(f"  {col}: {n}/{len(tuning_df)}")
    for col in ['reliability', 'self_force', 'partner_force']:
        n = (tuning_df[col].abs() >= EFFECT_THRESHOLD_CONTINUOUS).sum()
        print(f"  {col}: {n}/{len(tuning_df)}")

    plot_tuning_heatmap(tuning_df, agent, fig_save_path, NEURAL_TAG)

tuning_A.to_pickle(data_save_path + f"unit_tuning_agentA_{NEURAL_TAG}.pkl")
tuning_B.to_pickle(data_save_path + f"unit_tuning_agentB_{NEURAL_TAG}.pkl")
print(f"\nSaved unit_tuning_agentA/B_{NEURAL_TAG}.pkl")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

STEPS_PER_SECOND = 10


# --- outlier removal, based on y (metric) values ---
def filter_outliers_by_y(x, y, method='iqr', k=1.5):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(y) < 4:
        return x, y, 0

    if method == 'iqr':
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        keep = (y >= lo) & (y <= hi)
    elif method == 'zscore':
        mean_y, std_y = y.mean(), y.std()
        if std_y == 0:
            keep = np.ones_like(y, dtype=bool)
        else:
            keep = np.abs(y - mean_y) <= k * std_y
    else:
        raise ValueError(f"unknown method: {method}")

    n_removed = (~keep).sum()
    return x[keep], y[keep], n_removed


# --- reuse from the real-data pipeline (same definitions) ---
def time_normalize_trajectory(trace, n_bins=100):
    L, n_components = trace.shape
    if L < 2:
        return np.full((n_bins, n_components), np.nan)
    orig_t = np.linspace(0, 1, L)
    new_t = np.linspace(0, 1, n_bins)
    out = np.zeros((n_bins, n_components))
    for c in range(n_components):
        f = interp1d(orig_t, trace[:, c], kind='linear')
        out[:, c] = f(new_t)
    return out


def smooth_pc_trajectory(traj_norm, method='gaussian', sigma=1.5, window=5):
    n_bins, n_components = traj_norm.shape
    smoothed = np.zeros_like(traj_norm)
    for c in range(n_components):
        if method == 'gaussian':
            smoothed[:, c] = gaussian_filter1d(traj_norm[:, c], sigma=sigma, mode='nearest')
        else:
            kernel = np.ones(window) / window
            smoothed[:, c] = np.convolve(traj_norm[:, c], kernel, mode='same')
    return smoothed


def compute_trajectory_metrics(traj, local_window=10):
    """
    traj: (n_bins, n_components) time-normalized (and typically smoothed) trajectory.
    Returns curvature, speed, AND tortuosity (global scalars + local profiles).
    """
    n_bins, n_components = traj.shape

    v = np.gradient(traj, axis=0)
    a = np.gradient(v, axis=0)

    v_norm2 = np.sum(v**2, axis=1)
    a_norm2 = np.sum(a**2, axis=1)
    v_dot_a = np.sum(v * a, axis=1)

    numerator = np.sqrt(np.clip(v_norm2 * a_norm2 - v_dot_a**2, a_min=0, a_max=None))
    denom = v_norm2**1.5
    denom[denom == 0] = np.nan
    pointwise_curvature = numerator / denom

    speed = np.sqrt(v_norm2)

    # --- global metrics ---
    global_curvature = np.nanmean(pointwise_curvature)
    global_speed = np.nanmean(speed)

    step_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc_length = np.sum(step_lengths)
    endpoint_dist = np.linalg.norm(traj[-1] - traj[0])
    global_tortuosity = arc_length / endpoint_dist if endpoint_dist > 0 else np.nan

    # --- local (moving window) metrics ---
    half_w = local_window // 2
    local_curvature = np.full(n_bins, np.nan)
    local_speed = np.full(n_bins, np.nan)
    local_tortuosity = np.full(n_bins, np.nan)

    for t in range(n_bins):
        lo, hi = max(0, t - half_w), min(n_bins, t + half_w + 1)
        if hi - lo < 3:
            continue
        local_curvature[t] = np.nanmean(pointwise_curvature[lo:hi])
        local_speed[t] = np.nanmean(speed[lo:hi])

        seg = traj[lo:hi]
        seg_arc = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
        seg_endpoint = np.linalg.norm(seg[-1] - seg[0])
        local_tortuosity[t] = seg_arc / seg_endpoint if seg_endpoint > 0 else np.nan

    return {
        'global_curvature': global_curvature,
        'global_speed': global_speed,
        'global_tortuosity': global_tortuosity,
        'local_curvature': local_curvature,
        'local_speed': local_speed,
        'local_tortuosity': local_tortuosity,
    }


def run_pca_on_hidden_trace(hidden_array, n_components=3, standardize=True):
    X = hidden_array.copy()
    if standardize:
        mean_ = X.mean(axis=0, keepdims=True)
        std_ = X.std(axis=0, keepdims=True)
        std_[std_ == 0] = 1.0
        X = (X - mean_) / std_
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)
    print(f"PCA explained variance ratio (PC1-{n_components}): {pca.explained_variance_ratio_}")
    return pcs, pca


def extract_interpull_trials(log_df, pcs, agent):
    pull_col = f'{agent}_pull'
    log_df = log_df.reset_index(drop=True)
    pull_idx = log_df.index[log_df[pull_col] == 1].to_numpy()

    trials = []
    for i in range(1, len(pull_idx)):
        prev_idx, this_idx = pull_idx[i-1], pull_idx[i]
        if log_df.loc[prev_idx, 'condition_id'] != log_df.loc[this_idx, 'condition_id']:
            continue
        trace = pcs[prev_idx + 1: this_idx + 1]
        if trace.shape[0] < 2:
            continue
        trials.append({
            'trial_start_idx': prev_idx + 1,
            'trial_end_idx': this_idx,
            'condition_id': log_df.loc[this_idx, 'condition_id'],
            'reliability': log_df.loc[this_idx, 'reliability'],
            'force_A': log_df.loc[this_idx, 'force_A'],
            'force_B': log_df.loc[this_idx, 'force_B'],
            'trace': trace,
        })
    print(f"Agent {agent}: extracted {len(trials)} inter-pull trials "
          f"(dropped first-pull-per-condition and any cross-condition trials)")
    return trials


def build_trial_metrics_df(trials, agent, n_bins=100, local_window=10,
                            smooth=True, smooth_sigma=1.5, steps_per_second=STEPS_PER_SECOND):
    rows = []
    traj_list = []
    self_force_col = 'force_A' if agent == 'A' else 'force_B'
    partner_force_col = 'force_B' if agent == 'A' else 'force_A'

    for tr in trials:
        raw_len_steps = tr['trace'].shape[0]
        ipi_sec = raw_len_steps / steps_per_second

        traj_norm = time_normalize_trajectory(tr['trace'], n_bins=n_bins)
        if np.isnan(traj_norm).any():
            continue
        if smooth:
            traj_norm = smooth_pc_trajectory(traj_norm, sigma=smooth_sigma)

        m = compute_trajectory_metrics(traj_norm, local_window=local_window)
        rows.append({
            'condition_id': tr['condition_id'],
            'reliability': tr['reliability'],
            'self_force': tr[self_force_col],
            'partner_force': tr[partner_force_col],
            'ipi': ipi_sec,
            'traj_norm': traj_norm,
            'global_curvature': m['global_curvature'],
            'global_speed': m['global_speed'],
            'global_tortuosity': m['global_tortuosity'],
            'mean_local_curvature': np.nanmean(m['local_curvature']),
            'mean_local_speed': np.nanmean(m['local_speed']),
            'mean_local_tortuosity': np.nanmean(m['local_tortuosity']),
        })
        traj_list.append(traj_norm)

    df = pd.DataFrame(rows)
    if len(traj_list) == 0:
        df['dist_from_grandmean'] = []
        return df

    grand_mean_traj = np.nanmean(np.stack(traj_list, axis=0), axis=0)
    dist_col = []
    for traj_norm in traj_list:
        per_bin_dist = np.linalg.norm(traj_norm - grand_mean_traj, axis=1)
        dist_col.append(np.nanmean(per_bin_dist))
    df['dist_from_grandmean'] = dist_col

    return df


def compute_trial_level_correlations(trial_metrics_df,
                                      predictor_vars=('reliability', 'self_force', 'partner_force', 'ipi'),
                                      metric_vars=('global_curvature', 'global_speed', 'global_tortuosity',
                                                   'mean_local_curvature', 'mean_local_speed',
                                                   'mean_local_tortuosity', 'dist_from_grandmean')):
    rows = []
    for pvar in predictor_vars:
        for mvar in metric_vars:
            x = trial_metrics_df[pvar].values.astype(float)
            y = trial_metrics_df[mvar].values.astype(float)
            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < 3 or np.std(x[valid]) == 0:
                r, p, n = np.nan, np.nan, valid.sum()
            else:
                r, p = pearsonr(x[valid], y[valid])
                n = valid.sum()
            rows.append({'predictor': pvar, 'metric_var': mvar, 'r': r, 'p': p, 'n_trials': n})
    return pd.DataFrame(rows)


def compute_condition_level_correlations(trial_metrics_df,
                                          predictor_vars=('reliability', 'self_force', 'partner_force', 'ipi'),
                                          metric_vars=('global_curvature', 'global_speed', 'global_tortuosity',
                                                       'mean_local_curvature', 'mean_local_speed',
                                                       'mean_local_tortuosity', 'dist_from_grandmean'),
                                          outlier_method='iqr', outlier_k=1.5):
    agg_dict = {}
    for p in predictor_vars:
        agg_dict[p] = 'mean' if p == 'ipi' else 'first'
    for m in metric_vars:
        agg_dict[m] = 'mean'

    cond_means = trial_metrics_df.groupby('condition_id').agg(agg_dict).reset_index()

    rows = []
    for pvar in predictor_vars:
        for mvar in metric_vars:
            x_all = cond_means[pvar].values.astype(float)
            y_all = cond_means[mvar].values.astype(float)
            valid = ~np.isnan(x_all) & ~np.isnan(y_all)
            x_all, y_all = x_all[valid], y_all[valid]

            x_clean, y_clean, n_removed = filter_outliers_by_y(
                x_all, y_all, method=outlier_method, k=outlier_k)

            if len(x_clean) < 3 or np.std(x_clean) == 0:
                r, p, n = np.nan, np.nan, len(x_clean)
            else:
                r, p = pearsonr(x_clean, y_clean)
                n = len(x_clean)

            rows.append({'predictor': pvar, 'metric_var': mvar, 'r': r, 'p': p,
                         'n_conditions': n, 'n_removed_outliers': n_removed})
    return pd.DataFrame(rows), cond_means


def plot_predictor_metric_scatter_AB_sidebyside(cond_means_A, cond_means_B,
                                                 predictor_vars=('reliability', 'self_force', 'partner_force', 'ipi'),
                                                 metric_vars=('global_curvature', 'global_speed', 'global_tortuosity',
                                                              'mean_local_curvature', 'mean_local_speed',
                                                              'mean_local_tortuosity', 'dist_from_grandmean'),
                                                 agent_colors={'A': '#C73E1D', 'B': '#2E86AB'},
                                                 outlier_method='iqr', outlier_k=1.5,
                                                 figsize_per_panel=(2.6, 2.6)):
    nrows = len(predictor_vars)
    ncols = len(metric_vars) * 2

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0]*ncols, figsize_per_panel[1]*nrows),
                              squeeze=False)

    for i, pvar in enumerate(predictor_vars):
        for j, mvar in enumerate(metric_vars):
            ax_A = axes[i, j*2]
            ax_B = axes[i, j*2 + 1]

            for ax, cond_means, agent in [(ax_A, cond_means_A, 'A'), (ax_B, cond_means_B, 'B')]:
                x_all = cond_means[pvar].values.astype(float)
                y_all = cond_means[mvar].values.astype(float)
                valid = ~np.isnan(x_all) & ~np.isnan(y_all)
                x_all, y_all = x_all[valid], y_all[valid]

                x, y, n_removed = filter_outliers_by_y(x_all, y_all, method=outlier_method, k=outlier_k)
                color = agent_colors.get(agent, 'gray')

                ax.scatter(x, y, s=14, alpha=0.7, color=color)
                if n_removed > 0:
                    removed_mask = ~np.isin(y_all, y)
                    ax.scatter(x_all[removed_mask], y_all[removed_mask],
                               s=20, marker='x', color='gray', alpha=0.5)

                if len(x) >= 3 and np.std(x) > 0:
                    r, p = pearsonr(x, y)
                    z = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 50)
                    ax.plot(xs, np.polyval(z, xs), color=color, linewidth=1.2)
                    label = f'r={r:.2f}\np={p:.3f}'
                    if n_removed > 0:
                        label += f'\n(-{n_removed} out.)'
                    ax.text(0.95, 0.92, label, transform=ax.transAxes,
                            ha='right', va='top', fontsize=6, color=color,
                            fontweight='bold' if p < 0.05 else 'normal')

                xlabel = 'IPI (s)' if pvar == 'ipi' else pvar.replace('_', ' ')
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel(mvar.replace('_', ' '), fontsize=8)
                ax.set_title(f'Agent {agent}', fontsize=8.5, color=color)

    fig.suptitle('Condition-level: force/reliability/IPI vs. PC-trajectory geometry\n'
                  '(A left, B right; y-axis outliers removed, gray x = removed)', fontsize=11)
    fig.tight_layout()
    return fig, axes


# =================================================================
# USAGE -- both agents
# =================================================================
pcs_A, pca_A = run_pca_on_hidden_trace(hidden_A, n_components=3)
pcs_B, pca_B = run_pca_on_hidden_trace(hidden_B, n_components=3)

trials_A = extract_interpull_trials(neural_log, pcs_A, agent='A')
trials_B = extract_interpull_trials(neural_log, pcs_B, agent='B')

trial_metrics_A = build_trial_metrics_df(trials_A, agent='A', n_bins=100, smooth_sigma=2.5)
trial_metrics_B = build_trial_metrics_df(trials_B, agent='B', n_bins=100, smooth_sigma=2.5)

predictor_vars = ('reliability', 'self_force', 'partner_force', 'ipi')
metric_vars = ('global_curvature', 'global_speed', 'global_tortuosity',
               'mean_local_curvature', 'mean_local_speed', 'mean_local_tortuosity',
               'dist_from_grandmean')

trial_corr_A = compute_trial_level_correlations(trial_metrics_A, predictor_vars, metric_vars)
trial_corr_B = compute_trial_level_correlations(trial_metrics_B, predictor_vars, metric_vars)

cond_corr_A, cond_means_A = compute_condition_level_correlations(
    trial_metrics_A, predictor_vars, metric_vars, outlier_method='iqr', outlier_k=2.5)
cond_corr_B, cond_means_B = compute_condition_level_correlations(
    trial_metrics_B, predictor_vars, metric_vars, outlier_method='iqr', outlier_k=2.5)

print("=== Agent A condition-level correlations (outliers removed) ===")
print(cond_corr_A)
print("\n=== Agent B condition-level correlations (outliers removed) ===")
print(cond_corr_B)

fig_scatter, axes_scatter = plot_predictor_metric_scatter_AB_sidebyside(
    cond_means_A, cond_means_B, predictor_vars=predictor_vars, metric_vars=metric_vars,
    outlier_method='iqr', outlier_k=2.5)
fig_scatter.savefig(fig_save_path + f"pcatraj_metrics_vs_force_reliability_ipi_AB_sidebyside_cleaned_{NEURAL_TAG}.pdf")


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_3d_pc_trajectories_AB_sharedaxis(trial_metrics_A, trial_metrics_B, color_by='self_force',
                                           min_n=3, figsize=(13, 6), title=None):
    """
    One figure, two 3D subplots side by side: Agent A (left), Agent B (right).
    X/Y/Z axis ranges (PC1/PC2/PC3) are computed jointly across both agents
    and applied to both panels, so visual differences in spread reflect real
    differences, not independent auto-scaling. Color scale (color_by) is
    also shared across both agents. Axis labels are explicit on both panels.

    NOTE: A and B have separate PCA fits (different hidden-state traces), so
    "PC1"/"PC2"/"PC3" are not the same computed direction for both agents --
    the shared ranges make magnitude/spread comparable, not axis meaning.
    """
    agent_dfs = {'A': trial_metrics_A, 'B': trial_metrics_B}

    cond_summaries = {}
    all_traj_points = []
    for agent, df in agent_dfs.items():
        cond_groups = df.groupby('condition_id')
        summary = cond_groups.agg({color_by: 'first', 'traj_norm': 'count'}).rename(
            columns={'traj_norm': 'n_trials'})
        summary = summary[summary['n_trials'] >= min_n].sort_values(color_by)
        cond_summaries[agent] = summary

        for cond_id in summary.index:
            sub = df[df['condition_id'] == cond_id]
            stacked = np.stack(sub['traj_norm'].values, axis=0)
            mean_traj = np.nanmean(stacked, axis=0)  # (n_bins, 3)
            all_traj_points.append(mean_traj)

    if len(all_traj_points) == 0:
        print(f"No conditions with >= {min_n} trials in either agent, nothing to plot")
        return None, None

    all_points = np.concatenate(all_traj_points, axis=0)
    x_lim = (all_points[:, 0].min(), all_points[:, 0].max())
    y_lim = (all_points[:, 1].min(), all_points[:, 1].max())
    z_lim = (all_points[:, 2].min(), all_points[:, 2].max())

    color_vals_all = np.concatenate([cond_summaries[a][color_by].values for a in agent_dfs])
    norm = plt.Normalize(vmin=color_vals_all.min(), vmax=color_vals_all.max())
    cmap = cm.viridis

    fig = plt.figure(figsize=figsize)
    axes = {}

    for k, agent in enumerate(['A', 'B']):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        axes[agent] = ax
        df = agent_dfs[agent]
        summary = cond_summaries[agent]

        for cond_id, row in summary.iterrows():
            sub = df[df['condition_id'] == cond_id]
            stacked = np.stack(sub['traj_norm'].values, axis=0)
            mean_traj = np.nanmean(stacked, axis=0)

            color = cmap(norm(row[color_by]))
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                     color=color, linewidth=1.5)
            ax.scatter(*mean_traj[0], color=color, marker='o', s=25)   # trial start
            ax.scatter(*mean_traj[-1], color=color, marker='X', s=40)  # pull time

        ax.set_xlabel('PC1', fontsize=11, labelpad=8)
        ax.set_ylabel('PC2', fontsize=11, labelpad=8)
        ax.set_zlabel('PC3', fontsize=11, labelpad=8)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_title(f'Agent {agent}', fontsize=12)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes.values()), shrink=0.6, label=color_by, pad=0.08)

    fig.suptitle(title or f'Inter-pull PC trajectories by {color_by}: Agent A (left) vs Agent B (right)',
                 fontsize=13)
    return fig, axes


# --- usage: one combined figure per color_by variable ---
fig_force, axes_force = plot_3d_pc_trajectories_AB_sharedaxis(
    trial_metrics_A, trial_metrics_B, color_by='self_force', min_n=3,
    title='Inter-pull PC trajectories by self force')
fig_force.savefig(fig_save_path + f"pcatraj_3d_by_selfforce_AB_sharedaxis_{NEURAL_TAG}.pdf")

fig_rel, axes_rel = plot_3d_pc_trajectories_AB_sharedaxis(
    trial_metrics_A, trial_metrics_B, color_by='reliability', min_n=3,
    title='Inter-pull PC trajectories by reliability')
fig_rel.savefig(fig_save_path + f"pcatraj_3d_by_reliability_AB_sharedaxis_{NEURAL_TAG}.pdf")


# In[ ]:


#################################################################
# FULL SWEEP: both training orders x all training seeds x both
# fixed_agent settings x multiple testing seeds, over a
# reliability x force grid, capturing BOTH behavior and neural
# (hidden-state) data for every combination via the existing
# run_neural_eval_with_cache pipeline (cell 20).
#
# Skips any combination whose training checkpoint doesn't exist yet
# (rather than crashing the whole sweep), and skips any eval that
# was already cached (via force_rerun=False). The summary table is
# saved after EVERY combination (not just at the end), so if the job
# is interrupted, you keep a record of everything completed so far.
#################################################################
import os
import itertools
import pandas as pd

# --- sweep configuration ---
SWEEP_ORDERS = ['A_then_B', 'B_then_A']
SWEEP_TRAIN_SEEDS = (0, 1, 2, 3, 4, 5)
SWEEP_FIXED_AGENTS = ['A', 'B']
SWEEP_TEST_SEEDS = (0, 1, 2, 3, 4, 5)   # 5 testing seeds

SWEEP_RELIABILITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SWEEP_FORCE_LEVELS = [100, 200, 300, 400, 500, 600, 700, 800, 900]
SWEEP_STEPS_PER_CONDITION = 3000   # matches the neural-logging default from cell 20

SWEEP_PHASE = 'phase2'   # final, fully-trained checkpoint (seen both effort phases)
SWEEP_FORCE_RERUN = 0    # 0 = skip anything already cached; 1 = redo everything

SWEEP_SUMMARY_PKL = data_save_path + "sweep_summary_ABxTrainSeedxFixedAgentxTestSeed.pkl"
SWEEP_SUMMARY_CSV = data_save_path + "sweep_summary_ABxTrainSeedxFixedAgentxTestSeed.csv"

sweep_summary = []


def save_summary_now(summary_list):
    """Overwrite the summary files with everything completed so far."""
    df = pd.DataFrame(summary_list)
    df.to_pickle(SWEEP_SUMMARY_PKL)
    df.to_csv(SWEEP_SUMMARY_CSV, index=False)
    return df


sweep_combos = list(itertools.product(SWEEP_ORDERS, SWEEP_TRAIN_SEEDS,
                                       SWEEP_FIXED_AGENTS, SWEEP_TEST_SEEDS))
print(f"Total sweep combinations: {len(sweep_combos)} "
      f"({len(SWEEP_ORDERS)} orders x {len(SWEEP_TRAIN_SEEDS)} train seeds x "
      f"{len(SWEEP_FIXED_AGENTS)} fixed_agent x {len(SWEEP_TEST_SEEDS)} test seeds)\n")

for combo_i, (order, train_seed, fixed_agent, test_seed) in enumerate(sweep_combos, start=1):
    print(f"\n[{combo_i}/{len(sweep_combos)}] order={order}, train_seed={train_seed}, "
          f"fixed_agent={fixed_agent}, test_seed={test_seed}")

    agent_A_path = data_save_path + f"agent_A_{order}_{CHECKPOINT_SET}_{SWEEP_PHASE}_seed{train_seed}.pth"
    agent_B_path = data_save_path + f"agent_B_{order}_{CHECKPOINT_SET}_{SWEEP_PHASE}_seed{train_seed}.pth"

    # --- skip combinations whose training checkpoint doesn't exist yet ---
    if not (os.path.exists(agent_A_path) and os.path.exists(agent_B_path)):
        print(f"  MISSING checkpoint(s) -- skipping "
              f"(expected {os.path.basename(agent_A_path)} / {os.path.basename(agent_B_path)})")
        sweep_summary.append({
            'order': order, 'train_seed': train_seed, 'fixed_agent': fixed_agent,
            'test_seed': test_seed, 'eval_tag': None, 'status': 'missing_checkpoint',
            'agent_A_path': agent_A_path, 'agent_B_path': agent_B_path,
        })
        save_summary_now(sweep_summary)
        continue

    base_tag = derive_eval_tag(agent_A_path, agent_B_path, fixed_agent)
    eval_tag = f"{base_tag}_testseed{test_seed}_sweep"

    try:
        neural_log, hidden_A, hidden_B = run_neural_eval_with_cache(
            agent_A_path, agent_B_path,
            SWEEP_RELIABILITY_LEVELS, SWEEP_FORCE_LEVELS,
            fixed_agent=fixed_agent, steps_per_condition=SWEEP_STEPS_PER_CONDITION,
            seed=test_seed, data_save_path=data_save_path, eval_tag=eval_tag,
            force_rerun=bool(SWEEP_FORCE_RERUN))

        sweep_summary.append({
            'order': order, 'train_seed': train_seed, 'fixed_agent': fixed_agent,
            'test_seed': test_seed, 'eval_tag': eval_tag, 'status': 'success',
            'agent_A_path': agent_A_path, 'agent_B_path': agent_B_path,
            'n_log_rows': len(neural_log), 'hidden_A_shape': hidden_A.shape,
            'hidden_B_shape': hidden_B.shape,
            'n_conditions': len(SWEEP_RELIABILITY_LEVELS) * len(SWEEP_FORCE_LEVELS),
        })

    except Exception as e:
        print(f"  ERROR during eval: {e}")
        sweep_summary.append({
            'order': order, 'train_seed': train_seed, 'fixed_agent': fixed_agent,
            'test_seed': test_seed, 'eval_tag': eval_tag, 'status': f'error: {e}',
            'agent_A_path': agent_A_path, 'agent_B_path': agent_B_path,
        })

    # --- save the summary after EVERY combination, not just at the end ---
    save_summary_now(sweep_summary)
    print(f"  Summary saved ({len(sweep_summary)}/{len(sweep_combos)} combos recorded so far)")

# --- final save + report (redundant with the per-combo saves, but harmless) ---
sweep_summary_df = save_summary_now(sweep_summary)

print("\n=== SWEEP COMPLETE ===")
print(sweep_summary_df['status'].value_counts())
print(f"\nSaved summary table: {SWEEP_SUMMARY_PKL} / {SWEEP_SUMMARY_CSV}")
print(f"Use eval_tag values from this table to load cached results later, e.g.:")
print(f"  neural_log = pd.read_pickle(data_save_path + f'neural_log_{{eval_tag}}.pkl')")
print(f"  hidden_A = np.load(data_save_path + f'neural_hiddenA_{{eval_tag}}.npy')")
print(f"  hidden_B = np.load(data_save_path + f'neural_hiddenB_{{eval_tag}}.npy')")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




