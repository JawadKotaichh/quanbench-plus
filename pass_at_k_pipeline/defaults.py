import numpy as np
import networkx as nx

DEFAULT_MODELS = [
    "openai/gpt-4.1",
    "openai/gpt-5.1",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-2.5-flash",
    "google/gemini-3-pro-preview",
    "meta-llama/llama-4-maverick",
    "qwen/qwen-2.5-7b-instruct",
    "alfredpros/codellama-7b-instruct-solidity",
    "z-ai/glm-4.7",
    "minimax/minimax-m2.1",
    "moonshotai/kimi-k2-thinking",
]

NUMBER_OF_SHOTS = 1000

TASK4_GRAPH = [[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]]
G = nx.Graph()
G.add_edges_from(TASK4_GRAPH)

GLOBAL_INPUTS = {
    "04": [
        G,
        [((25 * np.pi) / 54) for i in range(5)],
        [((25 * np.pi) / 54) for i in range(5)],
    ],
    "29": [1, 0],
    "39": [((25 * np.pi) / 54), ((25 * np.pi) / 54)],
    "40": [((25 * np.pi) / 54) for i in range(8)],
    "41": [((25 * np.pi) / 54) for i in range(8)],
    "42": [((25 * np.pi) / 54), ((25 * np.pi) / 54), ((25 * np.pi) / 54)],
}
