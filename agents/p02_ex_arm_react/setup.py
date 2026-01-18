from setuptools import setup, find_packages

setup(
    name="p02_ex_arm_react",
    version="0.1.0",
    description="ReAct型フィジカルAIエージェント",
    packages=find_packages(),
    package_dir={
        'my_agent': 'my_agent',  # パッケージ名を保持
    },
    # namespace_packagesを使わず、各エージェントを独立させる
    py_modules=[],
    install_requires=[
        "langchain",
        "langchain-openai",
        "langgraph",
        "python-dotenv",
        "pybullet",
        "numpy",
        "torch",
        "transformers",
        "pillow",
        "opencv-python",
    ],
)
