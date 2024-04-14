''' 第一步：预训练 LLM 模型 '''
# 构建初始化的 LLM 模型

LLM_model = LLM(config_args)
# 收集语料库 corpus
dataset = Dataset()
dataloader = Dataloader(dataset)
# 预训练 LLM 模型
for (prompt, target) in dataloader:
  """
  prompt shape: [batch_size, input_length, 1]
  target shape; [batch_size, output_length, 1]
  response shape: [batch_size, output_length, vocab_dim]
  """
  # 这里的 LLM 模型输出 2 部分：
  #   第一部分：和普通的 LLM 一样，即 next token 的概率
  #   第二部分：输出预测的 value，是为了第三步的 PPO 算法服务
  response, _ = LLM_model(prompt)
  loss = CEloss(response, target)  # cross-entropy loss
  optimizer.zero_grad()
  loss.back()
  optimizer.step()
# optional: SFT
sft_dataset = Dataset(collect_from_human)
sft_dataloader = Dataloader(sft_dataset)
for (prompt, target) in sft_dataloader:
  """
  prompt shape: [batch_size, input_length, 1]
  target shape; [batch_size, output_length, 1]
  response shape: [batch_size, output_length, vocab_dim]
  """
  response = LLM_model(prompt)
  loss = CEloss(response, target)  # cross-entropy loss
  optimizer.zero_grad()
  loss.back()
  optimizer.step()
'''第二步：预训练 Reward 模型'''
Reward_model = LLM(reward_config_args)  # reward model 和 LLM model 架构是一致的
prompts = collect()  # 收集 prompts
reward_train_data = []
for prompt in prompts:
  # 通过设置不同的随机数等方式来使得 LLM model 对同一个 prompt 输出不同的 response
  multi_response = LLM_model(prompt, seed = different_seed)
  ranked_response = rank_by_human(multi_response)  # 通过人工对不同的 response 进行排序
  response_score = convert_to_score(ranked_response)  # 通过 elo 等方式将排序转化为 score
  reward_train_data.append({'ranked_response': ranked_response,
                            'response_score': response_score,
                            'prompt': prompt})
for responses in reward_train_data:
  prompt = responses['prompt']
  ranked_response = responses['ranked_response']
  response_score = responses['response_score']
  for response, score in zip(ranked_response, response_score):
    """
    prompt shape: [input_length, 1]
    response shape: [output_length, 1]
    score shape: [1]
    """
    score = score.repeat(response.shape[0], 0)  # score shape: [output_length, 1]
    predict = Reward_model(concat(prompt, response))
    # prompt 预测的 score 没有用
    # predict_score shape: [output_length, 1]
    predict_score = predict[prompt.shape[0]:]
    loss = MSEloss(predict_score, score)
    optimizer.zero_grad()
    loss.back()
    optimizer.step()
'''第三步：使用 PPO 算法微调 LLM 模型'''
prompts = collect()  # 收集 prompts
for prompt in prompts:
  # 初始化 advantage, rewards, values
  advantage = torch.zeros(max_trajects_length)  # max_trajects_length = output_length
  rewards = torch.zeros(max_trajects_length)
  values = torch.zeros(max_trajects_length)
  old_trajectory = [{'s0': prompt, 'a0': torch.randn(vocab_dim)}]
  # 也可以进一步将其通过预训练的 LLM 模型进行初始化
  old_trajectory = []
  for t in range(max_trajects_length):
      """
      a_t shape: [vocab_dim]
      value_t shape: [1]
      """
      a_t, value_t = LLM_model(s_t)
      reward_t = Reward_model(concat(s_t, torch,argmax(a_t)))
      rewards[t] = reward_t
      values[t] = value_t
      old_trajectory.append({'st': s_t, 'at': a_t})
  for k in range(epochs):  # 每个 prompt 迭代 epochs 次
    """
    根据上一次迭代的 trajectory 计算 advantage
    """
    last_advantage = 0
    last_value = 0
    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(max_trajects_length)):
      delta = rewards[t] + gamma * last_value - values[t]
      last_advantage = delta + gamma * gae_lambda * last_advantage
      advantages[t] = last_advantage
      last_value = values[t]
    # PPO 算法迭代
    s_t = prompt
    new_trajectory = []
    for t in range(max_trajects_length):
      """
      a_t shape: [vocab_dim]
      value_t shape: [1]
      """
      a_t, value_t = LLM_model(s_t)
      reward_t = Reward_model(concat(s_t, torch,argmax(a_t)))
      rewards[t] = reward_t
      values[t] = value_t
      new_trajectory.append({'st': s_t, 'at': a_t})
      advantage_t = advantage[t]
      gate = (1 + epsilon) * advantage_t if advantage_t > 0 else (1 - epsilon) * advantage_t
      loss = -min((a_t / old_trajectory[t]['at']).mean() * advantage_t, gate)
      optimizer.zero_grad()
      loss.back()
      optimizer.step()
    old_trajectory = new_trajectory