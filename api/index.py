from flask import Flask, request, jsonify, send_file
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from flask_cors import CORS 
import pathlib
from supabase import create_client, Client
from dotenv import load_dotenv
import base64
import requests
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Load environment variables and initialize Supabase client
load_dotenv()
url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = Flask(__name__)
CORS(app) 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@app.route('/')
def documentation():
    """Serve the API documentation HTML page."""
    doc_path = pathlib.Path(__file__).parent / "documentation.html"
    return send_file(doc_path)

@app.route('/api/summarize', methods=['POST'])
def summarize_image():
    """
    Summarize the content visible on a whiteboard in an image.
    
    Expects a JSON payload with:
    - imageUrl (string): URL of the image to analyze
    
    Returns:
    - JSON with 'summary' field containing a brief description of the whiteboard content
    - The summary starts with 'The whiteboard shows a:' and is limited to 50 words
    - Returns error 400 if imageUrl is missing
    - Returns error 500 if processing fails
    """
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
    """
    Extract and format content from a whiteboard image, preserving structure.
    
    Expects a JSON payload with:
    - imageUrl (string): URL of the image to analyze
    
    Returns:
    - JSON with 'tableContent' field containing the extracted and formatted content
    - Returns 'No content detected in the image' if no content is visible
    - Returns error 400 if imageUrl is missing
    - Returns error 500 if processing fails
    """
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
    """
    Classify content by assigning the most appropriate labels from a given list.
    
    Expects a JSON payload with:
    - content (string): The text content to classify
    - availableLabels (list): A list of possible labels to choose from
    
    Returns:
    - JSON with 'labels' field containing a list of assigned labels
    - Only returns labels that are in the provided availableLabels list
    - Returns an empty list if no labels are relevant
    - Returns error 400 if content or availableLabels is missing
    - Returns error 500 if processing fails
    """
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

@app.route('/api/session/<id>/summary', methods=['GET'])
def getSessionSummary(id):
    """
    Generate a summary for a whiteboard session based on all its images.
    
    Path Parameters:
    - id: The ID of the session to summarize
    
    Returns:
    - JSON with 'summary' field containing the AI-generated session summary
    - Returns error 404 if session is not found
    - Returns error 400 if session ID is invalid
    - Returns error 500 if there's an issue with processing
    """


    try:
        logging.info(f"Fetching session with ID: {id}")
        # Fetch session to verify it exists
        session = supabase.table("Session").select("*").eq("id", id).execute()
        
        if len(session.data) == 0:
            logging.warning(f"Session with ID {id} not found")
            return jsonify({"error": "session not found"}), 404
            
        logging.info(f"Session with ID {id} found. Fetching associated board states.")
        # Fetch all board states associated with this session through session_state join
        board_states = supabase.table("session_state") \
            .select("Board_State!inner(id, imageUrl, description)") \
            .eq("session", id) \
            .execute()
            
        # Extract board states from the nested structure
        board_states_data = [item["Board_State"] for item in board_states.data]
        logging.info(f"Found {len(board_states_data)} board states for session ID {id}")
        
        # Filter to only include board states with image URLs
        image_board_states = [state for state in board_states_data if state.get("imageUrl")]
        logging.info(f"Filtered to {len(image_board_states)} board states with image URLs")
        
        if not image_board_states:
            logging.info(f"No images found for session ID {id}")
            return jsonify({"summary": "No images found for this session"}), 200
            
        # Create message content with the prompt
        prompt = """
        **Prompt:**

        You are an AI tasked with analyzing images of a whiteboard captured during a session. Your objective is to provide a comprehensive summary of the content presented on the whiteboard. 

        1. **Image Analysis**: Carefully examine the provided images for any text, diagrams, bullet points, or drawings. Identify and recognize all relevant elements.
           
        2. **Content Extraction**: Extract key phrases, concepts, and any structured information from the whiteboard images. Pay attention to headings, subheadings, lists, and any visual aids that might convey important information.

        3. **Context Understanding**: Based on the content extracted, infer the context of the session. Determine the main topics discussed and the relationships between different ideas presented.

        4. **Summary Construction**: Produce a comprehensive and well-organized summary that includes:
           - A brief introduction outlining the session's objective.
           - A detailed breakdown of key points, grouped by topic or theme.
           - Any conclusions or action items that were highlighted during the session.
           - Notable conclusions, action items, or decisions documented.

        5. **Clarity and Coherence**: Ensure the summary is clear, coherent, and well-structured. Use bullet points or numbered lists where appropriate to improve readability.
        The summary can be as long as needed to fully capture the content—do not shorten or omit important details for the sake of brevity.
        6. **Output Guidelines**:
        Your output should be a comprehensive summary of the whiteboard session.
        Provide the summary directly, without introductory phrases like “Here is the summary...”

        and always add the following to the end of the summary:
        "This summary was generated by an AI model and may not capture all details accurately."
        """
        
        message_content = [
            {
                "type": "text",
                "text": prompt.strip()
            }
        ]
        
        # Add each image URL directly to the message content
        for state in image_board_states:
            image_url = state.get("imageUrl")
            if image_url:
                logging.info(f"Adding image URL to message content: {image_url}")
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
        
        # If we couldn't find any valid image URLs
        if len(message_content) <= 1:
            logging.warning(f"No valid image URLs found for session ID {id}")
            return jsonify({"summary": "Could not process any images for this session"}), 200
        
        # Call the AI model
        logging.info(f"Invoking AI model for session ID {id}")
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        logging.info(f"AI model response received for session ID {id}")
        
        # Update the session with the summary if it doesn't already have one
        if not session.data[0].get("summary"):
            logging.info(f"Updating session ID {id} with generated summary")
            supabase.table("Session").update({"summary": response.content}).eq("id", id).execute()
        
        return jsonify({"summary": response.content}), 200
    
    except ValueError as e:
        logging.error(f"Invalid session ID: {id}. Error: {str(e)}")
        return jsonify({"error": "invalid session id"}), 400
    except Exception as e:
        logging.error(f"An error occurred while generating session summary for ID {id}: {str(e)}")
        return jsonify({"error": "an error occurred while generating session summary"}), 500

if __name__ == '__main__':
    app.run(debug=True)