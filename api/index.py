from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@app.route('/api/summarize', methods=['POST'])
def summarize_image():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({'error': 'Missing imageUrl'}), 400

    try:
        prompt = (
            "Can you summarize what's written on the whiteboard, "
            "start your sentence with 'The whiteboard shows a:' "
            "and make the response no bigger than 50 words."
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ])

        response = llm.invoke([message])
        return jsonify({'summary': response.content})

    except Exception as e:
        print("Error:", e)
        return jsonify({'summary': 'Unable to generate summary'}), 500

@app.route('/api/extract-table', methods=['POST'])
def extract_table():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({'error': 'Missing imageUrl'}), 400

    try:
        prompt = (
            "Extract and format the content from this whiteboard image. "
            "Preserve the structure and formatting of the content. "
            "If there is no content visible, respond with 'No content detected in the image.'"
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
        ])

        response = llm.invoke([message])
        return jsonify({'tableContent': response.content})

    except Exception as e:
        print("Error:", e)
        return jsonify({'tableContent': 'Unable to extract table content'}), 500

@app.route('/api/classify', methods=['POST'])
def classify_content():
    data = request.get_json()
    content = data.get('content')
    available_labels = data.get('availableLabels', [])

    if not content:
        return jsonify({'error': 'Missing content'}), 400
    
    if not available_labels:
        return jsonify({'error': 'Missing available labels'}), 400

    try:
        labels_str = ", ".join(available_labels)
        prompt = (
            f"Given the following content, assign the most appropriate labels from this list: {labels_str}.\n\n"
            f"Content: {content}\n\n"
            "Return ONLY the labels as a comma-separated list. Choose only labels that are truly relevant. "
            "If none are relevant, return an empty list."
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
        ])

        response = llm.invoke([message])
        
        # Parse the response to get labels
        response_text = response.content.strip()
        if not response_text:
            assigned_labels = []
        else:
            # Parse the comma-separated list and clean up each label
            assigned_labels = [label.strip() for label in response_text.split(',')]
            # Only include labels that are in the available labels list
            assigned_labels = [label for label in assigned_labels if label in available_labels]

        return jsonify({'labels': assigned_labels})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Unable to classify content', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)