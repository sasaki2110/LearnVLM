from setuptools import setup, find_packages

setup(
    name="p01_ex_arm",
    version="0.1.0",
    description="LangGraph agent for VLM robot bridge",
    packages=find_packages(),
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
    ],
)
