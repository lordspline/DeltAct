# Core imports
import instructor
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
import base64
from PIL import Image
import supervision as sv
import numpy as np
import json
import time
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

# Configuration
SAMPLE_SIZES = [1, 5, 10, 20, 50]
NUM_TEST_TASKS = 100  # Number of tasks to evaluate
TARGET_W = 700
TARGET_H = 1100

# Response Models for structured outputs
class Response(BaseModel):
    current_webpage_analysis: str
    previous_action_analysis: str
    screenshot_details_analysis: str
    next_action_analysis: str
    proposed_action_element: int
    proposed_action_op: str
    proposed_action_value: str
    validity_check: str
    final_action_element: int
    final_action_op: str
    final_action_value: str

# Helper functions for image processing
def process_and_encode_image(screenshot, ranked_candidates):
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    candidate_label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_position=sv.Position.BOTTOM_LEFT,
        text_scale=0.5,
        text_color=sv.Color.white(),
        color=sv.Color.black(),
        text_thickness=1
    )
    
    candidate_detections = convert_elements2detections(ranked_candidates)
    candidate_labels = [str(i) for i in range(len(candidate_detections))]
    
    annotated_image = bounding_box_annotator.annotate(
        scene=np.array(screenshot), 
        detections=candidate_detections
    )
    annotated_image = candidate_label_annotator.annotate(
        scene=annotated_image, 
        detections=candidate_detections, 
        labels=candidate_labels
    )
    screenshot = Image.fromarray(annotated_image)
    screenshot = screenshot.resize((TARGET_W, TARGET_H))
    screenshot.save("temp.jpg")
    with open("temp.jpg", "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Model interaction functions
def get_gpt4_response(prompt, image_base64, client):
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=1024
        )
        return response
    except Exception as e:
        print(f"Error with GPT-4: {e}")
        return None

def get_claude_response(prompt, image_base64, client, model="claude-3-sonnet-20240229"):
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }]
        )
        return response
    except Exception as e:
        print(f"Error with Claude: {e}")
        return None

class Strategy(Enum):
    SIMPLE_SAMPLING = "simple_sampling"
    MCTS = "mcts"

@dataclass
class MCTSNode:
    action: Optional[tuple] = None  # (element_id, operation, value)
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: List[tuple] = None

    def __post_init__(self):
        self.children = []
        self.untried_actions = []

# Main experiment runner
class ScalingExperiment:
    def __init__(self):
        self.openai_client = OpenAI()
        self.anthropic_client = Anthropic()
        self.dataset = load_dataset("osunlp/Multimodal-Mind2Web")
        self.results = []
        self.exploration_constant = 1.414  # UCT exploration parameter
        
    def run_single_task(self, task, model, num_samples, strategy=Strategy.SIMPLE_SAMPLING):
        if strategy == Strategy.SIMPLE_SAMPLING:
            return self.run_simple_sampling(task, model, num_samples)
        else:
            return self.run_mcts(task, model, num_samples)
    
    def run_simple_sampling(self, task, model, num_samples):
        """Original sampling strategy"""
        results = []
        for _ in range(num_samples):
            image_base64 = process_and_encode_image(task["screenshot"], task["ranked_candidates"])
            
            if model.startswith("gpt"):
                response = get_gpt4_response(task["prompt"], image_base64, self.openai_client)
            else:
                response = get_claude_response(task["prompt"], image_base64, self.anthropic_client, model)
            
            if response:
                results.append(self.evaluate_response(response, task))
                
        return max(results) if results else 0

    def run_mcts(self, task, model, num_samples):
        """MCTS-based task solving"""
        root = MCTSNode()
        root.untried_actions = self.get_action_space(task)
        
        for _ in range(num_samples):
            node = self.select_node(root)
            if node.untried_actions:
                child = self.expand(node, task)
                simulation_result = self.simulate(child, task, model)
                self.backpropagate(child, simulation_result)
            
        best_child = max(root.children, key=lambda c: c.visits)
        return float(best_child.value / best_child.visits if best_child.visits > 0 else 0)

    def evaluate_response(self, response, task):
        # Implement evaluation logic
        # Return 1 for success, 0 for failure
        pass
    
    def run_experiment(self):
        models = [
            "gpt-4-vision-preview",
            "claude-3-sonnet-20240229"
        ]
        
        strategies = [Strategy.SIMPLE_SAMPLING, Strategy.MCTS]
        
        results_df = pd.DataFrame(columns=[
            'model', 'strategy', 'num_samples', 'success_rate', 'cost'
        ])
        
        test_tasks = self.dataset["test_website"][:NUM_TEST_TASKS]
        
        for model in models:
            for strategy in strategies:
                for n_samples in SAMPLE_SIZES:
                    successes = []
                    costs = []
                    
                    for task in tqdm(test_tasks):
                        success = self.run_single_task(
                            task, 
                            model, 
                            n_samples, 
                            strategy
                        )
                        cost = self.calculate_cost(model, n_samples)
                        
                        successes.append(success)
                        costs.append(cost)
                    
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'model': [model],
                        'strategy': [strategy.value],
                        'num_samples': [n_samples],
                        'success_rate': [np.mean(successes)],
                        'cost': [np.mean(costs)]
                    })], ignore_index=True)
                    
        return results_df
    
    def calculate_cost(self, model, num_samples):
        # Updated cost calculation with just the two models
        cost_per_call = {
            "gpt-4-vision-preview": 0.13,  # GPT-4V cost
            "claude-3-sonnet-20240229": 0.017  # Claude 3.5 Sonnet cost
        }
        return cost_per_call[model] * num_samples

    def get_action_space(self, task):
        """Generate possible actions from the ranked candidates"""
        actions = []
        for candidate in task["ranked_candidates"]:
            # Basic click action for all elements
            actions.append((
                candidate["rank"],
                "CLICK",
                ""
            ))
            
            # Add TYPE action for input elements
            if candidate["tag"].lower() == "input":
                actions.append((
                    candidate["rank"],
                    "TYPE",
                    ""  # Will be filled during simulation
                ))
                
            # Add SELECT action for select elements
            if candidate["tag"].lower() == "select":
                actions.append((
                    candidate["rank"],
                    "SELECT",
                    ""  # Will be filled during simulation
                ))
        
        return actions

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT"""
        while node.untried_actions == [] and node.children != []:
            node = self.select_best_child(node)
        return node

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """Use UCT formula to select best child"""
        def uct_value(child: MCTSNode) -> float:
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.value / child.visits
            exploration = self.exploration_constant * np.sqrt(
                np.log(node.visits) / child.visits
            )
            return exploitation + exploration
        
        return max(node.children, key=uct_value)

    def expand(self, node: MCTSNode, task) -> MCTSNode:
        """Expand the selected node with a new child"""
        if not node.untried_actions:
            return node
            
        action = node.untried_actions.pop()
        child = MCTSNode(
            action=action,
            parent=node
        )
        child.untried_actions = self.get_action_space(task)
        node.children.append(child)
        return child

    def simulate(self, node: MCTSNode, task, model) -> float:
        """Simulate from the current node to get a result"""
        action_sequence = self.get_action_sequence(node)
        modified_task = task.copy()
        modified_task["actions_so_far"] = action_sequence
        
        image_base64 = process_and_encode_image(
            task["screenshot"], 
            task["ranked_candidates"]
        )
        
        if model.startswith("gpt"):
            response = get_gpt4_response(
                self.create_prompt(modified_task), 
                image_base64, 
                self.openai_client
            )
        else:
            response = get_claude_response(
                self.create_prompt(modified_task), 
                image_base64, 
                self.anthropic_client,
                model
            )
        
        if response:
            return float(self.evaluate_response(response, task))
        return 0.0

    def backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate the result up the tree"""
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    # Helper methods
    def get_action_sequence(self, node: MCTSNode) -> List[str]:
        """Get the sequence of actions leading to this node"""
        sequence = []
        current = node
        while current.parent is not None:
            if current.action:
                element_id, operation, value = current.action
                action_str = f"{operation} element {element_id}"
                if value:
                    action_str += f" with value '{value}'"
                sequence.append(action_str)
            current = current.parent
        return list(reversed(sequence))

    def create_prompt(self, task) -> str:
        """Create a prompt for the model including action history"""
        prompt = f"""Task: {task['task']}

Previous actions:
{chr(10).join(task['actions_so_far'])}

Analyze the current webpage state and suggest the next action to take."""
        return prompt

# Analysis functions
def plot_scaling_curves(results_df):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    for model in results_df['model'].unique():
        for strategy in results_df['strategy'].unique():
            data = results_df[
                (results_df['model'] == model) & 
                (results_df['strategy'] == strategy)
            ]
            plt.plot(
                data['num_samples'], 
                data['success_rate'], 
                label=f"{model}-{strategy}", 
                marker='o'
            )
    
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Success Rate')
    plt.title('Scaling of Success Rate with Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('scaling_curves.png', bbox_inches='tight')
    
def analyze_cost_efficiency(results_df):
    efficiency_df = results_df.copy()
    efficiency_df['success_per_dollar'] = efficiency_df['success_rate'] / efficiency_df['cost']
    return efficiency_df

# Main execution
if __name__ == "__main__":
    experiment = ScalingExperiment()
    results = experiment.run_experiment()
    
    # Save raw results
    results.to_csv('scaling_results.csv', index=False)
    
    # Generate plots and analysis
    plot_scaling_curves(results)
    efficiency_analysis = analyze_cost_efficiency(results)
    efficiency_analysis.to_csv('cost_efficiency.csv', index=False)
    print("\nCost Efficiency Analysis:")
    print(efficiency_analysis)