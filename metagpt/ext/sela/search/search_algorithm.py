import numpy as np

from metagpt.ext.sela.search.tree_search import BaseTreeSearch, Node


class Greedy(BaseTreeSearch):
    def best_child(self):
        if len(self.children) == 0:
            return self.root_node
        all_children = [child for children in self.children.values() for child in children]
        return max(all_children, key=lambda x: x.normalized_reward.get("dev_score", 0))


class Random(BaseTreeSearch):
    def best_child(self):
        if len(self.children) == 0:
            return self.root_node
        all_children = [child for children in self.children.values() for child in children]
        return np.random.choice(all_children)


class MCTS(BaseTreeSearch):
    # modified uct by x
    def best_child(self):
        def hybrid_uct(node: Node):
            # 混合价值计算
            if node.visited > 0:
                real_value = node.value / node.visited
                virtual_weight = self.c_virtual_visit / (node.visited + self.c_virtual_visit)
                hybrid_val = virtual_weight * node.llm_value + (1 - virtual_weight) * real_value
                n_visits = node.visited
            else:
                hybrid_val = node.llm_value
                n_visits = self.c_virtual_visit  # 避免除零

            # 动态探索系数（预测不确定性越大，探索倾向越高）
            uncertainty = 1 
            # uncertainty = 1 / (1 + np.abs(node.llm_value))  # 示例调节逻辑
            explore_term = self.c_explore * uncertainty * np.sqrt(np.log(node.parent.visited + 1e-6) / n_visits)
            
            return hybrid_val + explore_term

        def uct(node: Node):
            n_visits = node.visited if node.visited else self.c_unvisited
            avg_value = node.avg_value() if node.visited else node.value / self.c_unvisited
            return avg_value + self.c_explore * np.sqrt(np.log(node.parent.visited) / n_visits)

        if len(self.children) == 0:
            return self.root_node
        all_children = [child for children in self.children.values() for child in children]
        
        return max(all_children, key=hybrid_uct)