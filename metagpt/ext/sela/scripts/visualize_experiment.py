import networkx as nx

from metagpt.ext.sela.evaluation.visualize_mcts import (
    build_tree_recursive,
    visualize_tree,
)
from metagpt.ext.sela.search.tree_search import create_initial_state, initialize_di_root_node
from metagpt.ext.sela.search.search_algorithm import MCTS, Greedy, Random
from metagpt.ext.sela.run_experiment import get_args
from metagpt.ext.sela.utils import DATA_CONFIG
import os



def find_latest_directory(base_path, prefix):
    latest_dir = None
    latest_time = -1

    # 遍历指定路径下的所有文件夹
    all_dirs = []
    for entry in os.scandir(base_path):
        if entry.is_dir() and entry.name.startswith(prefix):
            all_dirs.append(entry.path)
            # 获取文件夹的创建时间
            creation_time = os.path.getctime(entry.path)
            # 检查是否是最新的文件夹
            is_contatin_file = os.path.isfile(os.path.join(entry.path, "node_order.json"))
            if creation_time > latest_time and is_contatin_file:
                latest_time = creation_time
                latest_dir = entry.path

    return all_dirs, latest_dir

if __name__ == "__main__":
    args = get_args()
    data_config = DATA_CONFIG
    state = create_initial_state(args.task, 0, data_config, args=args, visualized_mode=True)
    role, node = initialize_di_root_node(state)

    if not args.node_dir:
    # 自动获取输入task_name的最新日期的node_dir 来进行打印
        base_path = data_config["role_dir"]
        prefix = args.task
        _, state["node_dir"] = find_latest_directory(base_path, prefix)
    args.node_dir = state["node_dir"]

    print(f"loading tree from {args.node_dir}")
    suffix = os.path.basename(args.node_dir)

    mcts = MCTS(
        root_node=node,
        max_depth=5,
        use_fixed_insights=False,
    )

    mcts.load_tree()
    mcts.load_node_order()
    root = mcts.root_node
    node_order = mcts.node_order
    G = nx.DiGraph()
    build_tree_recursive(G, "0", root, node_order)
    

    visualize_tree(G, save_path=f"results/{suffix}-tree.png")
    print(f"Visualized Tree saved in results/{suffix}-tree.png")
