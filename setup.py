from setuptools import setup, find_packages

setup(
    name="software-ai-team",
    version="1.0.0",
    author="Sy Truong",
    description="An autonomous multi-agent software engineering team.",
    packages=find_packages(),
    install_requires=[
        "langchain-google-genai",
        "langgraph",
        "langchain-core"
    ],
    entry_points={
        "console_scripts": [
            "ai-team=software_team.cli:main",
        ],
    },
)