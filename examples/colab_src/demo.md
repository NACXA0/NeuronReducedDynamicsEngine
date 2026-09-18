# %% [markdown]
# # NRDE A1 Demo (Colab)
#
# 果蝇接口 Demo —— **不**主张有意义行为（RFC-001 D5）。
#
# 安装：

# %%
# !pip install -q "neuron-reduced-dynamics-engine[fly]"

# %% [markdown]
# 拉取种子并跑 50 步 headless：

# %%
from nrde import demo
from nrde.fetch import fetch_preset

report = fetch_preset("fly", tier="seed")
print("fetch ok:", report["ok"])
result = demo("flygym", steps=50, headless=True)
print(result)
