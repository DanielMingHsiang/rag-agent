import logging
from pathlib import Path


Path("./log").mkdir(parents=True, exist_ok=True)


logging.basicConfig(level=logging.INFO)

retrieval_graph_logger = logging.getLogger("retrieval")
retrieval_graph_logger.addHandler(
    logging.StreamHandler(
        open(file="./log/retrieval_graph.log", mode="a", encoding="utf-8")
    )
)


react_agent_logger = logging.getLogger("react_agent")
react_agent_logger.addHandler(
    logging.StreamHandler(
        open(file="./log/react_agent.log", mode="a", encoding="utf-8")
    )
)

kb_retrieval_agent_logger = logging.getLogger("kb_retrieval_agent")
kb_retrieval_agent_logger.addHandler(
    logging.StreamHandler(
        open(file="./log/kb_retrieval_agent.log", mode="a", encoding="utf-8")
    )
)
