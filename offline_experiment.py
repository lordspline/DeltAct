import json
import jsonlines
import os
from dotenv import load_dotenv

load_dotenv(".ienv.local")

# from src.demo_utils.inference_engine import OpenaiEngine
from inference_engine import OpenaiEngine
from prompts import generate_prompt


generation_model = OpenaiEngine(rate_limit=-1, model="gpt-4-turbo-2024-04-09")

exp_split = "text_choice"

base_dir = os.path.dirname(os.path.abspath(__file__))
source_data_path = "../../data/examples/textual_choice"
source_data_path = os.path.join(base_dir, source_data_path)

for action_file in os.listdir(source_data_path):
    if action_file.startswith(".") or not os.path.isdir(
        os.path.join(source_data_path, action_file)
    ):
        continue

    print(f"Start testing: {action_file}")

    if os.path.exists(
        os.path.join(source_data_path, action_file, f"prediction-{exp_split}.jsonl")
    ):
        print("Prediction already exist")
        continue
    query_meta_data = []
    with open(os.path.join(source_data_path, action_file, "queries.jsonl")) as reader:
        for obj in reader:
            query_meta_data.append(json.loads(obj))
    predictions = []
    for query_id, query in enumerate(query_meta_data):
        print("-" * 10)
        print(os.path.splitext(os.path.basename(action_file))[0] + "-" + str(query_id))
        image_path = query["image_path"] + "/" + str(query_id) + ".jpg"
        image_path = image_path.replace("../", "")
        image_path = image_path.replace("./", "")
        image_path = os.path.join(source_data_path, image_path)
        choices_input = None
        try:
            choices_input = query["choices"]
        except:
            pass
        prompt_list = generate_prompt(
            exp_split,
            task=query["confirmed_task"],
            previous=query["previous_actions"],
            choices=choices_input,
        )
        print("-" * 10)
        print(prompt_list[0])
        print(prompt_list[1])

        output0 = generation_model.generate(
            prompt=prompt_list, image_path=image_path, turn_number=0
        )
        print("#" * 10)
        print(output0)
        print("-" * 10)
        output1 = generation_model.generate(
            prompt=prompt_list, image_path=image_path, turn_number=1, ouput__0=output0
        )

        print(prompt_list[2])
        print("#" * 10)
        print(output1)

        output_list = [output0, output1]
        output_jsonl = dict(
            multichoice_id=query_id, gpt_output=output_list, prompt=prompt_list
        )
        predictions.append(output_jsonl)
    with jsonlines.open(
        os.path.join(source_data_path, action_file, f"prediction-{exp_split}.jsonl"),
        mode="w",
    ) as writer:
        writer.write_all(predictions)
