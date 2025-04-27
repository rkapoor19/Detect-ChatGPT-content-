from sentence_transformers import SentenceTransformer
import joblib
import re
import os

class LLMDetector:
    def __init__(self, model_path='models/model_checkpoint'):
        self.embedder = SentenceTransformer(f'{model_path}/embedder')
        self.classifier = joblib.load(f'{model_path}/classifier.joblib')

    def predict(self, text):
        embedding = self.embedder.encode([text])
        prediction = self.classifier.predict(embedding)
        probability = self.classifier.predict_proba(embedding)[0][1]
        return prediction[0], probability

    def split_into_sentences(self, text):
        # Improved sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    def color_for_probability(self, prob):
        # Return color code based on how "AI-ish" the sentence is
        if prob >= 0.9:
            return 'background-color: red; color: white;'
        elif prob >= 0.8:
            return 'background-color: orange;'
        elif prob >= 0.7:
            return 'background-color: yellow;'
        else:
            return ''  # No highlight if less than 0.7

    def highlight_sentences_html(self, text):
        sentences = self.split_into_sentences(text)
        highlighted = ''

        for sentence in sentences:
            if not sentence.strip():
                continue
            pred, prob = self.predict(sentence)
            style = self.color_for_probability(prob)
            if style:
                highlighted += f'<span style="{style}">{sentence}</span> '
            else:
                highlighted += f'{sentence} '
        return highlighted.strip()

    def save_highlighted_html(self, input_text, output_path='highlighted_output.html'):
        highlighted_text = self.highlight_sentences_html(input_text)

        html_template = f"""
        <html>
        <head><title>LLM Detector Output</title></head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2>Text with Highlighted ChatGPT-like Sentences</h2>
        <p>{highlighted_text}</p>
        <br><br>
        <p><strong>Legend:</strong></p>
        <ul>
          <li><span style="background-color: red; color: white;">Red</span>: Very strong ChatGPT signature (prob > 0.9)</li>
          <li><span style="background-color: orange;">Orange</span>: Strong AI signature (prob 0.8 - 0.9)</li>
          <li><span style="background-color: yellow;">Yellow</span>: Possible AI signature (prob 0.7 - 0.8)</li>
        </ul>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

        print(f"[+] Highlighted HTML saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    detector = LLMDetector()

    # Get text
    sample_text = input("Enter text to check: ")
    
    label, prob = detector.predict(sample_text)
    label_name = "ChatGPT" if label == 1 else "Human"
    print(f"\nOverall Prediction: {label_name} (Confidence: {prob:.2f})")

    # Save highlighted HTML
    detector.save_highlighted_html(sample_text, output_path='highlighted_output.html')
