# RL models from elegantrl
import torch
import numpy as np
from train.config import Arguments
from train.run import train_and_evaluate, init_agent

from drl_agents.agents import AgentDDPG, AgentPPO, AgentSAC, AgentTD3, AgentA2C

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "a2c": AgentA2C}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "a2c"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""


class DRLAgent:
    """Provides implementations for DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get train_results
    """

    def __init__(self, env, price_array, tech_array, env_params, if_log):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.env_params = env_params
        self.if_log = if_log

    def get_model(self, model_name, gpu_id, model_kwargs):

        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "if_train": False,
        }

        env = self.env(config=env_config,
                       env_params=self.env_params,
                       if_log=self.if_log)

        env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model = Arguments(agent=agent, env=env)
        
        # 支持多GPU设置
        if isinstance(gpu_id, (list, tuple)) and len(gpu_id) > 1:
            # 如果传入的是GPU ID列表，则设置为多GPU模式
            model.learner_gpus = gpu_id
            print(f"设置多GPU训练模式: {gpu_id}")
        else:
            # 单GPU模式
            model.learner_gpus = gpu_id

        if model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_time_gap"]
                
                # 添加worker_num参数支持
                if "worker_num" in model_kwargs:
                    model.worker_num = model_kwargs["worker_num"]
                    print(f"设置worker_num为{model.worker_num}个工作进程")
                else:
                    model.worker_num = 4  # 默认设置为4个工作进程
                    print("设置默认worker_num为4")
                    
                # 添加多GPU相关参数
                if isinstance(model.learner_gpus, (list, tuple)) and len(model.learner_gpus) > 1:
                    # 当使用多GPU时，需要考虑多进程训练参数
                    if "learner_num" in model_kwargs:
                        model.learner_num = model_kwargs["learner_num"]
                    else:
                        model.learner_num = len(model.learner_gpus)  # 默认等于GPU数量
                    print(f"设置learner_num为{model.learner_num}")
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        
        # 检查是否需要使用多GPU训练
        if isinstance(model.learner_gpus, list) and len(model.learner_gpus) > 1:
            print(f"\n使用多GPU训练：{model.learner_gpus}")
            from train.run import train_and_evaluate_mp
            train_and_evaluate_mp(model)
        else:
            from train.run import train_and_evaluate
            train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, gpu_id):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent = MODELS[model_name]
        environment.env_num = 1

        args = Arguments(agent=agent, env=environment)

        args.cwd = cwd
        args.net_dim = net_dimension
        # load agent
        try:
            agent = init_agent(args, gpu_id=gpu_id)
            act = agent.act
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)

        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)

                total_asset = (
                        environment.cash
                        + (
                                environment.price_array[environment.time] * environment.stocks
                        ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break
        print("\n Test Finished!")
        print("episode_return: ", episode_return - 1, '\n')
        return episode_total_assets
