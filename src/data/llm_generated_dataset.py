import json
from tqdm import tqdm

from config import SYSTEM_PROMPT, NUM_SAMPLES


def generate_dataset(generator, num_examples=NUM_SAMPLES):
    dataset = []
    for _ in tqdm(range(num_examples)):
        response = generator(
            SYSTEM_PROMPT,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )[0]["generated_text"]

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    dataset.append(obj)
                except json.JSONDecodeError:
                    pass
    return dataset


if __name__ == "__main__":

    from utils import load_model, save_dataset, get_device
    from config import MODEL_NAME

    DEVICE = get_device()

    generator = load_model(MODEL_NAME, DEVICE)
    data = generate_dataset(generator, num_examples=2)
    save_dataset(data)
