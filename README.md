# CS181-FinalProject-Liars-Bar

This project is a framework for playing the Liars Bar game with reinforcement learning (Deep Q Network), expectimax, and heuristic search.

## State Space
The state space consists of the following components:


## Key Components and Functions

### Game Logic
- **[`game.py`](game.py)**: Implements the core game logic.

### Reinforcement Learning Utilities
- **[`rl_utils.py`](rl_utils.py)**: Provides utility functions for encoding game states, managing action spaces, and extracting features for reinforcement learning agents.

#### Classes and Functions in `rl_utils.py`

##### `StateEncoder`
Encodes the game state into a format suitable for reinforcement learning agents.
- `encode_play_state(round_base_info, round_action_info, play_decision_info, hand, target_card, current_bullet) -> Tuple`: Encodes the state during the play phase.
- `encode_challenge_state(round_base_info, round_action_info, challenge_decision_info, challenging_player_performance, hand, target_card, current_bullet) -> Tuple`: Encodes the state during the challenge phase.
- `_encode_base_state(round_base_info, hand, target_card, current_bullet) -> Tuple`: Encodes the common base state shared by both play and challenge phases.
- `_encode_hand(hand, target_card) -> Dict`: Encodes the player's hand into counts of each card type.

##### `PlayActionTemplateSpace`
Defines a fixed template space for play actions.
- `__init__(max_hand_size: int = 5)`: Initializes the action space with a maximum hand size.
- `_build_templates() -> List[Tuple[int, int, int, int]]`: Builds all possible action templates based on the maximum hand size.
- `to_cards(tpl: Tuple[int, int, int, int]) -> List[str]`: Converts an action template into a list of cards.
- `is_legal(tpl: Tuple[int, int, int, int], hand: List[str]) -> bool`: Checks if a given action template is legal based on the player's hand.
- `n -> int`: Returns the total number of action templates.

##### `ActionDecoder`
Converts actions from the reinforcement learning agent into a format understandable by the game.
- `__init__(max_hand_size: int = 5)`: Initializes the decoder with a maximum hand size.
- `num_play_actions() -> int`: Returns the number of play actions in the action space.
- `get_legal_play_actions(hand: List[str]) -> List[int]`: Returns a list of legal action IDs based on the player's hand.
- `get_play_action_mask(hand: List[str]) -> np.ndarray`: Returns a mask indicating which actions are legal for the play phase.
- `decode_play_action(action_idx: int, hand: List[str]) -> Dict`: Decodes an action ID into the actual cards to be played.
- `num_total_actions() -> int`: Returns the total number of actions, including play and challenge actions.
- `get_total_action_mask(phase: str, hand: List[str]) -> np.ndarray`: Returns a mask for all actions (play and challenge) based on the current phase.

##### `FeatureExtractor`
Extracts feature vectors from the game state for use in reinforcement learning.
- `get_features(state: Tuple) -> np.ndarray`: Extracts a feature vector from the game state.
- `_encode_target_card(target_card: str) -> List[int]`: Encodes the target card as a one-hot vector.

### Reinforcement Learning
- **[`DQNAgent.py`](DQNAgent.py)**: Implements a Deep Q-Network (DQN) agent.
- input: use mask to represent the valid actions
  - use lazy initialization and replay buffer to store experiences


- **[`LinearQAgent.py`](LinearQAgent.py)**: Implements a linear Q-learning agen- `LinearQAgent.update(state, action, reward, next_state)`: Updates the Q-values.


- **[`rl_trainer.py`](rl_trainer.py)**: Manages the training process for RL agents.



- **[`network.py`](network.py)**: Defines neural network architectures for RL agents.
- `build_dqn(input_dim, output_dim)`: Builds a DQN model.

### Data Handling
- **[`game_record.py`](game_record.py)**: Handles game recording and replay.
- **[`json_convert.py`](json_convert.py)**: Converts game data to and from JSON format.


