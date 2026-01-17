from setuptools import setup, find_packages

setup(
    name="p00_sample",
    version="0.1.0",
    description="LangGraph sample agent",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-openai",
        "langgraph",
        "python-dotenv",
    ],
)
