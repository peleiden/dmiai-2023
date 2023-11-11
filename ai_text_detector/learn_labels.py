from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import requests
NUM_EXAMPLES = 1082
ALL_ZEROS_SCORE = 0.5295748613678373




def post_attempt():
    post_url = "https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue"
    post_headers = {"x-token": "aba10b0bc50e4ce6bdc180b2e05df4cc"}
    post_data = {
        "url": "http://82.211.207.131:6969/predict"
    }
    post_response = requests.post(post_url, headers=post_headers, json=post_data)
    return post_response.json()

def check_improved(uuid: str):
    while True:
        time.sleep(np.random.exponential(0.5))
        get_url = f"https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue/{uuid}/attempt"
        get_headers = {"x-token": "aba10b0bc50e4ce6bdc180b2e05df4cc"}

        get_response = requests.get(get_url, headers=get_headers)
        get_response_json = get_response.json()
        if get_response_json.get("finished_at"):
            return int(get_response_json["score"] > ALL_ZEROS_SCORE)


def main():
    already_checked = set()
    out = Path("true_labels.thingy")
    if out.exists():
        with open(out, "r") as file:
            already_checked = {int(line.split()[0]) for line in file.readlines()}

    for idx in tqdm(range(NUM_EXAMPLES)):
        if idx in already_checked:
            continue
        with open("idx_to_test.int", "w", encoding="utf-8") as file:
            file.write(str(idx))
        while True:
            time.sleep(np.random.exponential(1))
            response = post_attempt()
            if response.get("status") == "queued":
                uuid = response["queued_attempt_uuid"]
                break
            else:
                print("Attempt POST unsuccesful, retrying ...")
        label = check_improved(uuid)
        with open(out, "a") as file:
            file.write("%i %i\n" % (idx, label))


if __name__ == "__main__":
    main()
