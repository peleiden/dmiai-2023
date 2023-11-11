NUM_EXAMPLES = ...
ALL_ZEROS_SCORE = ...


def main():
    for idx in range(NUM_EXAMPLES):
        with open("idx_to_test.int", "w", encoding="utf-8") as file:
            file.write(str(idx))

if __name__ == "__main__":
    main()
