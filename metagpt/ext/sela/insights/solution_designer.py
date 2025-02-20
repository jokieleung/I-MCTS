import json
from metagpt.ext.sela.utils import clean_json_from_rsp, load_data_config, mcts_logger
from metagpt.llm import LLM

DATA_CONFIG = load_data_config()


DATASET_DESCRIPTION_SELA_PROMPT = """
# Dataset Description
{dataset}

# Dataset Metadata
{metadata}

# Dataset Head
{head}
"""

DATASET_DESCRIPTION_CUSTOM_PROMPT = """
# Dataset Description
{dataset_description}
"""

DATASET_INSIGHT_PROMPT = """
{description}

# Instruction
Propose insights to help improve the performance of the model on this dataset.
The insights should be proposed based on the dataset description with different task types.
Each task type should have at least 5 insights.
Make sure each method is diverse enough and can be implemented separately.
Be specific about models' choices, ensemble and tuning techniques, and preprocessing & feature engineering techniques.
Your model choices should be advanced enough to be helpful.

# Format
```json
[
    {{
        "task_type": "EDA",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Data Preprocessing",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Feature Engineering",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Model Training",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }}
]
```
"""


INSIGHT_PROPOSAL_PROMPT = """
You are an AI assistant tasked with analyzing a machine learning solution and proposing new insights to IMPORVE its performance. Given the current plan, solution code, Evaluation Metrics and performance on development score, suggest innovative approaches to ENHANCE and IMPORVE the performance.

# Current Plan:
{current_plan}

# Current Solution Code:
{solution_code}

# Current Evaluation Metrics:
{current_metrics}

# Current Performance on Development Set: 
{dev_score}

There may be some drawbacks in the Current Plan or Solution that leads to a lower development score.
Based on this information, please propose ONE new insight w.r.t. the following specific stage of the machine learning pipeline to IMPORVE the overall performance. Your insight should be specific, actionable, and have the potential to IMPORVE the model's performance.
{task_type}

NOTE that the following insights have been proposed before. Kindly AVOID suggesting an insight that is similar to them.
{cur_expansion_list}

Please strictly format your response as a JSON with the following structure:
```json
{{
    "task_type": "Data Preprocessing",
    "critic_feedback": "feedback and explanation on potential drawbacks in the Current Plan or Solution that leads to a lower development score"
    "insight": "ONE insight that specific, actionable, and have the potential to IMPORVE the model's performance."
}}
```
"""


KEY_DATASET_FEATURES = [
    "NumberOfClasses",
    "NumberOfFeatures",
    "NumberOfInstances",
    "NumberOfInstancesWithMissingValues",
    "NumberOfMissingValues",
    "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures",
]

TASK_TO_ID = {"EDA": 1, "Data Preprocessing": 2, "Feature Engineering": 3, "Model Training": 4, "Model Evaluation": 5}
ID_TO_TASK = {1: "EDA", 2: "Data Preprocessing", 3: "Feature Engineering", 4: "Model Training", 5: "Model Evaluation"}

class SolutionDesigner:
    data_dir: str = DATA_CONFIG["datasets_dir"]

    async def generate_solutions(self, dataset_info, dataset_name, save_analysis_pool=True):
        llm = LLM()
        if type(dataset_info) == dict:
            description_prompt = DATASET_DESCRIPTION_SELA_PROMPT.format(
                dataset=dataset_info["description"],
                metadata=self.metadata_builder(dataset_info["metadata"]),
                head=dataset_info["df_head"],
            )
        else:
            description_prompt = DATASET_DESCRIPTION_CUSTOM_PROMPT.format(dataset_description=dataset_info)
        context = DATASET_INSIGHT_PROMPT.format(description=description_prompt)
        rsp = await llm.aask(context)
        rsp = clean_json_from_rsp(rsp)
        analysis_pool = self.process_analysis_pool(json.loads(rsp))
        if save_analysis_pool:
            dataset_path = f"{self.data_dir}/{dataset_name}"
            self.save_analysis_pool(dataset_path, analysis_pool)
        return analysis_pool

    async def propose_new_insights(self, plan, solution, current_metrics, score, task_type, cur_expansion_list):
        llm = LLM()
        context = INSIGHT_PROPOSAL_PROMPT.format(current_plan=plan, solution_code=solution, current_metrics=current_metrics, dev_score=score,task_type=task_type, cur_expansion_list=cur_expansion_list)
        attempt = 0
        retries = 3
        while attempt < retries:
            try:
                rsp = await llm.aask(context)
                mcts_logger.log("MCTS", f"[I-MCTS]propose_new_insights:context {context}")
                mcts_logger.log("MCTS", f"[I-MCTS]propose_new_insights:rsp {rsp}")
                rsp = clean_json_from_rsp(rsp)
                mcts_logger.log("MCTS", f"[I-MCTS]propose_new_insights:clean rsp {rsp}")
                new_insights = self.my_process_generated_insight(json.loads(rsp))
                return new_insights
            except Exception as e:
                attempt += 1
                if attempt == retries:
                    raise e
                else:
                    print(f"Attempt {attempt} failed with error: {e}. Retrying...")

        # rsp = await llm.aask(context)
        # rsp = clean_json_from_rsp(rsp)
        # new_insights = self.process_analysis_pool(json.loads(rsp))
        # return new_insights

    def process_analysis_pool(self, insights_rsp):
        analysis_pool = []
        for task_type_insights in insights_rsp:
            task_type = task_type_insights["task_type"]
            for insight in task_type_insights["insights"]:
                analysis_pool.append({"Analysis": insight, "Category": task_type, "task_id": TASK_TO_ID[task_type]})
        return analysis_pool
        
    def process_generated_insight(self, insights_rsp):
        analysis_pool = []
        for task_type_insights in insights_rsp:
            task_type = task_type_insights["task_type"]
            insight = task_type_insights["insight"]
            critic_feedback = task_type_insights["critic_feedback"]
            analysis_pool.append({"Category": task_type, "critic_feedback": critic_feedback,"Analysis": insight, "task_id": TASK_TO_ID[task_type]})
        return analysis_pool

    def my_process_generated_insight(self, task_type_insights):
        task_type = task_type_insights["task_type"]
        insight = task_type_insights["insight"]
        critic_feedback = task_type_insights["critic_feedback"]
        ret = {"Category": task_type, "critic_feedback": critic_feedback,"Analysis": insight, "task_id": TASK_TO_ID[task_type]}
        return ret

    def metadata_builder(self, qualities):
        metadata = {}
        for key in KEY_DATASET_FEATURES:
            metadata[key] = qualities.get(key, "N/A")
        metadata_text = json.dumps(metadata, indent=4)
        return metadata_text

    def save_analysis_pool(self, dataset_path, analysis_pool):
        fpath = f"{dataset_path}/ds_analysis_pool.json"
        with open(fpath, "w") as file:
            json.dump(analysis_pool, file, indent=4)
