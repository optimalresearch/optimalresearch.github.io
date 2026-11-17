### Best run

#### Submission

[Astra’s submission](runs/astra.html#final) learns task performance only from trusted prompts, whose rewards contain no hack bonus. It also trains a detector to identify likely hacks that the original monitor misses. When either the monitor or detector flags an answer, the RL update applies a penalty that pushes the model to produce that answer less often. After training, the language model generates answers without consulting the detector.

The method uses monitor labels from ordinary prompts to help train the detector. It also compares flagged answers with higher-reward, unflagged answers to the same trusted prompt. It reads frozen base-model activations for the prompt and response through a small neural network, so its predictions can depend on context.

The RL update compares each answer’s trusted reward with the other answers to the same prompt, using a leave-one-out baseline. Applying that same subtraction to the safety penalty would cancel it whenever every answer incurs the same cost. The method instead applies the safety penalty without subtracting the group baseline. A weaker supervised loss reinforces high-reward answers that pass both checks, including examples from earlier training steps.

#### Development process

Astra’s early attempts to imitate high-reward trusted answers still produced hacking on sorting. This led the agent to train a detector on monitor labels and use its predictions to penalize suspected hacks during training.

The selected method combined a detector that considers both the prompt and answer with a safety penalty that remains effective when every sampled answer receives the same penalty. Longer evaluations improved dev results across all three environments. The agent then tested a more elaborate detector that learned its own representations and used separate validation prompts to check its predictions. That version did not improve the aggregate dev score, so the agent retained the earlier method.

### Across the runs

All six submissions avoid directly optimizing the hack bonus by learning task performance only from the small subset of trusted prompts. They leave the larger set of ordinary prompts unused for task learning, potentially giving up performance that a method could gain by using that feedback without reinforcing hacks.

To address the hacking that remains, the submissions use learned detectors, imitation of successful answers, preferences for shorter answers, or checkpoint selection on trusted validation prompts.

Astra and [Opus’s detector run](runs/opus-3.html#final) both learn from monitor labels and penalize additional answers that their detectors flag. The Opus agent introduced its detector after trusted-reward training still produced unflagged copies on sorting. The detector reduced hacking in the pilots, but the model continued to struggle with the sorting task. At final evaluation, both submissions achieved low hack rates, while Astra retained more task reward. The Opus model’s lower task reward shows why reducing hacking alone was not enough to match Astra’s score.

[Sol](runs/sol.html#final) combines trusted-reward RL with imitation of the shortest unflagged answer among those receiving the highest reward. Adding RL to its imitation baseline reduced observed sorting hacks while preserving a similar dev score. In [another Opus run](runs/opus-2.html#final), favoring shorter successful answers improved the pilot scores, but adding imitation reduced them.

[Opus’s strongest run](runs/opus-1.html#final) uses the original monitor, penalizes answer length on trusted prompts, and selects a sampling temperature using trusted rewards. Its final change restricted the length penalty to trusted prompts, but a shorter dev run left sorting undertrained. [Opus’s weakest run](runs/opus-4.html#final) instead selects checkpoints using trusted validation reward and monitor penalties. It retained task performance at final evaluation but lost much of its score to hacking. Selecting checkpoints with trusted rewards and monitor penalties did not prevent that model from learning hacks the monitor missed.

{{figure:ideas_landscape}}

{{figure:ideas_timeline}}
