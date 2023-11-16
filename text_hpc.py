import json
import os
import requests
import time
from glob import glob as glob  # glob

from pelutils import log, Parser
import torch.cuda.amp as amp

from ai_text_detector.training.eval import Classifier

ADDR_RECEIVE = "http://82.211.207.131:6969/to-hpc"
ADDR_SEND = "http://82.211.207.131:6969/from-hpc"

def run():
    log.info("Started app. Beginning to ping ...")
    while True:
        try:
            texts: list[str] = json.loads(requests.get(ADDR_RECEIVE).content)
        except requests.ConnectionError:
            continue
        if texts:
            log("Predicting reviews for %i reviews" % len(texts))
            with amp.autocast():
                preds = MODEL.predict(texts, BATCH_SIZE)
            with open("valid.json", "w") as f:
                json.dump(dict(preds=preds, reviews=texts), f, indent=4)
            requests.post(ADDR_SEND, json={ "answers": preds })

if __name__ == "__main__":
    location = Parser().parse_args().location
    log.configure("text-hpc.log")
    with log.log_errors:
        MODEL = Classifier(glob(os.path.join(location, "dfm-encoder-large-v1*-ai-detector")))
        BATCH_SIZE = 128
        run()
