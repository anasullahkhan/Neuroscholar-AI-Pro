class ResearchPaper:
    def __init__(self, title, text):
        self.title = title
        self.text = text

    def clean_text(self):
        cleaned = self.text.lower()
        cleaned = cleaned.replace("\n", " ")
        cleaned = cleaned.strip()
        return cleaned

    def word_count(self):
        words = self.clean_text().split()
        return len(words)

    def sentence_count(self):
        sentences = self.clean_text().split(".")

        valid_sentences = []
        for sentence in sentences:
            if sentence.strip() != "":
                valid_sentences.append(sentence)

        return len(valid_sentences)

    def generate_report(self):
        report = f"""
RESEARCH PAPER REPORT

Title: {self.title}

Cleaned Text:
{self.clean_text()}

Total Words: {self.word_count()}
Total Sentences: {self.sentence_count()}
"""
        return report


def read_text_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        return text

    except FileNotFoundError:
        print(f"Error: {filename} was not found.")
        return ""


def save_text_file(filename, text):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Report saved successfully to {filename}")


# Main program
text = read_text_file("sample.txt")

if text != "":
    paper = ResearchPaper("Neuroscience Sample Paper", text)

    report = paper.generate_report()

    print(report)

    save_text_file("report.txt", report)
else:
    print("No text to analyze.")