from __future__ import annotations

import asyncio
import json
import os

from pydantic import model_validator

from metagpt.actions.di.write_analysis_code import WriteAnalysisCode
from metagpt.const import SERDESER_PATH
from metagpt.ext.sela.utils import mcts_logger, save_notebook
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.schema import Message, Task, TaskResult
from metagpt.utils.common import CodeParser, write_json_file


EVALUATOR_SYSTEM_MSG = "As a Kaggle Grandmaster competing in a challenge, Your role is to evaluate the candidate solution for the given data science problem."


EVALUATION_CRETERIA = """
### Evaluation Criteria for AI Engineer's Execution Plan in Kaggle Data Science Competition  
Below is a scoring framework to evaluate the quality and feasibility of an execution plan. The total score is **100 points**, divided into **7 criteria**, each with defined scoring ranges and descriptions.  

---

#### **1. Problem Understanding (15 points)**  
- **15**: Clearly defines the competition goal, success metrics (e.g., AUC, RMSE), and constraints (e.g., runtime limits). Includes domain-specific nuances (e.g., business impact for sales prediction).  
- **10–14**: Adequate understanding but misses minor details (e.g., unclear evaluation metric implications).  
- **5–9**: Superficial analysis; overlooks key competition rules or data constraints.  
- **0–4**: Incorrect problem framing or missing critical objectives.  

---

#### **2. Data Preprocessing & Feature Engineering (20 points)**  
- **20**: Comprehensive plan addressing missing values, outliers, categorical encoding, and normalization. Proposes novel features (e.g., interaction terms, domain-specific transformations) and validates their impact.  
- **15–19**: Solid strategy but lacks innovation (e.g., standard scaling but no feature interactions).  
- **10–14**: Basic techniques (e.g., mean imputation, one-hot encoding) with gaps (e.g., no outlier handling).  
- **5–9**: Incomplete or naive methods (e.g., dropping all missing values without analysis).  
- **0–4**: No preprocessing/feature engineering or harmful approaches.  

---

#### **3. Model Selection & Validation Strategy (20 points)**  
- **20**: Justifies model choices (e.g., LightGBM for tabular data, NN for high dimensionality) and advanced techniques (e.g., stacking, automated hyperparameter tuning). Designs robust cross-validation aligned with competition rules (e.g., time-series splits for temporal data).  
- **15–19**: Reasonable models (e.g., XGBoost) and CV but lacks optimization (e.g., no Bayesian hyperparameter search).  
- **10–14**: Basic models (e.g., random forest) with weak validation (e.g., simple holdout).  
- **5–9**: Inappropriate models (e.g., CNNs for small tabular data) or validation leakage.  
- **0–4**: No clear model selection or validation plan.  

---

#### **4. Training & Optimization (15 points)**  
- **15**: Efficient resource use (e.g., GPU acceleration, parallel trials), advanced optimization (e.g., class imbalance handling, quantile loss for skewed targets), and time management.  
- **10–14**: Logical training workflow but suboptimal resource allocation (e.g., no early stopping).  
- **5–9**: Basic training loop with critical inefficiencies (e.g., no hyperparameter tuning).  
- **0–4**: Unworkable training strategy (e.g., no batch processing for large data).  

---

#### **5. Post-Processing & Interpretation (10 points)**  
- **10**: Ensemble methods (e.g., weighted blending), thorough error analysis, and model interpretation (e.g., SHAP values, feature importance).  
- **7–9**: Simple ensembling (e.g., averaging) and basic interpretation (e.g., feature importance plots).  
- **4–6**: Minimal post-processing (e.g., no calibration) or superficial analysis.  
- **0–3**: No post-processing or interpretation.  

---

#### **6. Documentation & Reproducibility (10 points)**  
- **10**: Clean, modular code structure with detailed documentation (e.g., Docker setup, dependency lists). Includes version control and experiment tracking (e.g., MLflow).  
- **7–9**: Readable code but limited documentation (e.g., no README).  
- **4–6**: Disorganized code with critical reproducibility gaps.  
- **0–3**: Undocumented or non-reproducible code.  

---

#### **7. Innovation & Practicality (10 points)**  
- **10**: Novel yet practical ideas (e.g., synthetic data for imbalance, custom loss functions). Balances creativity with competition constraints.  
- **7–9**: Minor innovations (e.g., new feature engineering) but overly complex in parts.  
- **4–6**: Generic approach with no novel elements.  
- **0–3**: Impractical or gimmicky methods (e.g., unnecessary deep learning).  

---

### **Scoring Scale**  
| Total Score | Grade          | Description                                                                 |  
|-------------|----------------|-----------------------------------------------------------------------------|  
| **90–100**  | Exceptional    | Well-structured, innovative, and executable plan. High chance of top ranks. |  
| **70–89**   | Strong         | Minor gaps but logically sound. Competitive but needs refinement.           |  
| **50–69**   | Average        | Basic approach with significant oversights (e.g., weak validation).         |  
| **<50**     | Insufficient   | Flawed or incomplete plan; unlikely to succeed.                             |  

---

### **Evaluation Guidance**  
- Prioritize **justification of choices** (e.g., why CatBoost over XGBoost?).  
- Reward **pragmatism** (e.g., focusing on feature engineering over overly complex models).  
- Penalize **ignoring competition constraints** (e.g., submission runtime limits).  
- Highlight **risks** (e.g., overfitting due to improper CV) and mitigation strategies.  

This framework ensures a balance between technical rigor, innovation, and practicality in tabular data competitions.
"""

EVALUATE_NODE_BY_LLM = """

# Evaluation Criteria
{evaluation_Criteria}

# Instruction:
Based on the above evaluation Criteria, Please give feedback and rating for the following solution. 

# Candidate Case

## User Requirement
{user_requirement}

## Candidate Solution
{candidate_plan}

# Format:
Please ensure your output strictly adheres to the following structure, ensure the "Total_Score" is a INT number":
```json
{{
    "evaluation_feedback": "Your evaluation feedback for the candidate solution based on the evaluation criteria. Please provide a detailed analysis of each criterion and explain why you gave the specific rating",
    "total_score": x
}}

"""


        # def visualize_node(self, previous_plans=None):
        #     role = load_role(self)
        #     node_id = self.id
        #     plans = role.planner.plan.tasks
        #     instruct_plans = [f"{i+1}. {task.instruction}" for i, task in enumerate(plans)]
        #     if previous_plans is not None:
        #         instruct_plans = [plan for plan, prev_plan in zip(instruct_plans, previous_plans) if plan != prev_plan]
        #     instruct_plans_text = "\n".join(instruct_plans)
        #     simulated = role.state_saved
        #     score = f"avg score: {self.avg_value()}, simulated score: {self.raw_reward}"
        #     num_visits = self.visited
        #     return NODE_TEMPLATE.format(
        #         id=node_id, plans=instruct_plans_text, simulated=simulated, score=score, num_visits=num_visits
        #     )

CODE_BLOCK_RESULT = """
## Execution Result:
{result}
"""
# CODE_BLOCK_RESULT = """
# ## Code:
# {code}

# ## Execution Result:
# {result}
# """

EXTRACT_SCORE_PROMPT = """
# Code Blocks
{code_block}
# Instruction:
Based on the code and execution result, please extract the **final scores** and return it as a dictionary.
If you cannot find the scores, please still return a dictionary with the keys 'train_score', 'dev_score', and 'test_score', and set the values to -1.

# Format:
```json
{{
    "train_score": x.x,
    "dev_score": x.x,
    "test_score": x.x
}}
```
"""


class TimeoutException(Exception):
    pass


def async_timeout():
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            try:
                result = await asyncio.wait_for(func(self, *args, **kwargs), timeout=self.role_timeout)
            except asyncio.TimeoutError:
                text = f"Function timed out after {self.role_timeout} seconds"
                mcts_logger.error(text)
                self.save_state()
                raise TimeoutException(text)
            return result

        return wrapper

    return decorator


class Experimenter(DataInterpreter):
    node_id: str = "0"
    start_task_id: int = 1
    state_saved: bool = False
    role_dir: str = SERDESER_PATH.joinpath("team", "environment", "roles", "Experimenter")
    role_timeout: int = 1000

    async def get_plan(self, goal):
        if not self.planner.plan.goal:
            # create initial plan 
            await self.planner.update_plan(goal=goal)
        

    def get_node_name(self):
        return f"Node-{self.node_id}"

    def get_next_instruction(self):
        mcts_logger.info(f"start_task_id: {self.start_task_id}")
        mcts_logger.info(f"len(self.planner.plan.tasks): {len(self.planner.plan.tasks)}")
        self.start_task_id = self.start_task_id if self.start_task_id < len(self.planner.plan.tasks) else len(self.planner.plan.tasks) - 1
        mcts_logger.info(f"self.planner.plan.tasks[self.start_task_id].instruction: {self.planner.plan.tasks[self.start_task_id].instruction}")
        return self.planner.plan.tasks[self.start_task_id].instruction

    def change_next_instruction(self, new_instruction):
        if new_instruction is not None:
            self.start_task_id = self.start_task_id if self.start_task_id < len(self.planner.plan.tasks) else len(self.planner.plan.tasks) - 1
            self.planner.plan.task_map[str(self.start_task_id)].instruction = new_instruction
            self.remap_tasks()

    def update_til_start_task(self, role: Experimenter, backward: bool = True):
        if backward:
            # make sure the previous task instructions are matched
            assert (
                self.start_task_id == role.start_task_id - 1
            ), f"start_task_id: {self.start_task_id}, role.start_task_id: {role.start_task_id}"
            for i in range(self.start_task_id):
                if (
                    self.planner.plan.task_map[str(self.start_task_id)].instruction
                    != role.planner.plan.task_map[str(self.start_task_id)].instruction
                ):
                    mcts_logger.info("Previous task instructions not matched")
                    self.remap_tasks()
                    return
            # copy new role's task (self.start_task_id) to current role
            self.planner.plan.task_map[str(self.start_task_id)] = role.planner.plan.task_map[
                str(self.start_task_id)
            ].model_copy()
            self.remap_tasks()

        else:
            assert (
                self.start_task_id == role.start_task_id + 1
            ), f"start_task_id: {self.start_task_id}, role.start_task_id: {role.start_task_id}"
            if int(role.planner.plan.current_task_id) > self.start_task_id:
                for i in range(role.start_task_id):
                    self.planner.plan.task_map[str(i)] = role.planner.plan.task_map[str(i)].model_copy()
            self.remap_tasks()

    async def get_score(self):
        score_dict = await self.llm_extract_score()
        score_dict["score"] = score_dict["dev_score"]
        return score_dict

    async def llm_virtual_score(self,user_requirement=""):
        plans = self.planner.plan.tasks
        instruct_plans = [f"{i+1}. {task.instruction}" for i, task in enumerate(plans)]
        instruct_plans_text = "\n".join(instruct_plans)
        rsp = await self.llm.aask(EVALUATE_NODE_BY_LLM.format(evaluation_Criteria=EVALUATION_CRETERIA, user_requirement=user_requirement, candidate_plan=instruct_plans_text))
        mcts_logger.info(f"The llm_virtual_score is: {rsp}")
        json_block = CodeParser.parse_code(block=None, text=rsp)
        score_dict = json.loads(json_block)
        return score_dict

    async def llm_extract_score(self):
        num_tasks = len(self.planner.plan.task_map)
        task_map = self.planner.plan.task_map
        result_list = [CODE_BLOCK_RESULT.format(result=task.result) for task in self.planner.plan.tasks if task.result != ""][-4:]
        code_block = "\n".join(result_list)
        rsp = await self.llm.aask(EXTRACT_SCORE_PROMPT.format(code_block=code_block, role="user"))
        mcts_logger.info(f"The llm extracted score is: {rsp}")
        json_block = CodeParser.parse_code(block=None, text=rsp)
        score_dict = json.loads(json_block)
        return score_dict

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "Interpreter":
        if self.planner.plan.goal != "":
            self.set_actions([WriteAnalysisCode])
            self._set_state(0)
            print("Plan already exists, skipping initialization.")
            return self
        print("Initializing plan and tool...")
        return super().set_plan_and_tool()

    async def _act_on_task(self, current_task: Task) -> TaskResult:
        """Useful in 'plan_and_act' mode. Wrap the output in a TaskResult for review and confirmation."""
        mcts_logger.info(f"The current_task is: {current_task}")
        code, result, is_success = await self._write_and_exec_code()
        task_result = TaskResult(code=code, result=result, is_success=is_success)
        if int(current_task.task_id) == self.start_task_id + 1:
            # fe_id = current_task.dependent_task_ids
            self.save_state()
            save_notebook(role=self, save_dir=self.role_dir, name=self.get_node_name(), save_to_depth=True)
        else:
            save_notebook(role=self, save_dir=self.role_dir, name=self.get_node_name())
        return task_result

    def get_solution(self):
        codes = [task.code for task in self.planner.plan.tasks]
        results = [task.result for task in self.planner.plan.tasks]
        return {"codes": codes, "results": results}

    def save_state(self, static_save=False):
        """
        attribute:
            state_saved - the state has been saved
        input:
            static_save - saving the state without changing the state_saved flag - used when a new role is created
        """
        if self.state_saved and not static_save:
            return
        if not static_save:
            self.state_saved = True
            mcts_logger.log("MCTS", f"Saving state at task {self.start_task_id}")
        else:
            mcts_logger.log("MCTS", "Static Saving")
        stg_path = self.role_dir
        name = self.get_node_name()
        role_path = os.path.join(stg_path, f"{name}.json")
        # save state as json file
        write_json_file(role_path, self.model_dump())

    def remap_tasks(self):
        self.planner.plan.tasks = [
            self.planner.plan.task_map[task_id] for task_id in sorted(self.planner.plan.task_map.keys())
        ]

    @async_timeout()
    async def run(self, with_message=None) -> Message | None:
        """Observe, and think and act based on the results of the observation"""
        if with_message == "continue":
            mcts_logger.info("Continue to run")
            self.rc.working_memory.clear()
            self.working_memory.clear()
            rsp = await self.react()
            self.set_todo(None)
            self.publish_message(rsp)
            return rsp
        return await super().run(with_message)
