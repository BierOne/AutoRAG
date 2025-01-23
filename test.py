from ragas.configuration.promptmaker import PromptMakerConfiguration
import yaml

# 假设你的YAML文件名为 'config.yaml'
with open('./ragas/configuration/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 打印读取的字典
print(config)


pm = PromptMakerConfiguration(config)
print(pm.cs)
print(pm.sampling(2))