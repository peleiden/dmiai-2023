import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv


def scrape_questions_answers(url):
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(url)
    time.sleep(3)
    qa_blocks = driver.find_elements(By.CLASS_NAME, "client-question-item")
    questions_answers = []
    for block in qa_blocks:
        question = block.find_element(By.CSS_SELECTOR, ".question .text").text.strip()
        answer = block.find_element(By.CSS_SELECTOR, ".answer .text").text.strip()
        if question and answer:
            questions_answers.append((question, answer, url))

    driver.quit()
    print(len(questions_answers))
    return questions_answers


def write_to_csv(data, filename):
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    url = sys.argv[1]
    qa_pairs = scrape_questions_answers(url)
    csv_filename = "questions_answers.csv"
    write_to_csv(qa_pairs, csv_filename)
