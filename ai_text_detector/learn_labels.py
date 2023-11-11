import requests
NUM_EXAMPLES = 1082
# ALL_ZEROS_SCORE = ...


def post_attempt():
    post_url = "https://cases.dmiai.dk/api/v1/usecases/ai-text-detector/validate/queue"
    post_headers = {"x-token": "aba10b0bc50e4ce6bdc180b2e05df4cc"}
    post_data = {
        "url": "http://82.211.207.131:6969/predict"
    }
    post_response = requests.post(post_url, headers=post_headers, json=post_data)
    return post_response.json()



def main():
    # for idx in range(NUM_EXAMPLES):
    #     with open("idx_to_test.int", "w", encoding="utf-8") as file:
    #         file.write(str(idx))
    post_attempt()
    breakpoint()


if __name__ == "__main__":
    main()
