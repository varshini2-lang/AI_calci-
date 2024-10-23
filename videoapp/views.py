from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
from .hand_detector import HandDetector  # Import your custom HandDetector
import google.generativeai as genai
from PIL import Image

# Initialize gemini
genai.configure(api_key="AIzaSyAJUcmudV_jencgPrXReZW36PhlgjsLlEw")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=0, detectionCon=0.75, minTrackCon=0.75)

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

response_text = None

def send_to_ai(model, image):
    global response_text
    response = model.generate_content(["solve this math problem", image])
    response_text = response.text if response else "No response from AI"

# Initialize variables
drawing = False
points = []  # Store points for drawing

def initialize_canvas(frame):
    return np.zeros_like(frame)

# Initialize canvas
_, frame = cap.read()
canvas = initialize_canvas(frame)

def video_stream():
    global drawing, points, canvas

    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        # Detect hands in the image
        img, hands = detector.findHands(img, draw=True, flipType=True)

        if hands:
            hand = hands[0]
            lmList = detector.findPosition(img)

            # Get the positions of the index and middle finger tips
            index_tip = lmList[8] if lmList else None
            thumb_tip = lmList[4] if lmList else None

            # Determine drawing state based on fingers up
            if index_tip is not None and thumb_tip is not None:
                fingers = detector.fingersUp(hand)
                if fingers[1] == 1 and fingers[2] == 0:  # Only index finger is up
                    drawing = True  # Set drawing to True
                    points.append(tuple(index_tip))

                elif fingers[1] == 1 and fingers[2] == 1:  # Both index and middle fingers are up
                    drawing = False
                    points = []  # Clear points to avoid connection

                elif fingers[0] == 1:  # Thumb is up
                    canvas = initialize_canvas(img)
                    points = []
                    drawing = False

                if drawing and len(points) > 1:
                    # Draw polyline on the canvas
                    cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=5)

                # Send to AI if the pinky is up
                if fingers[4] == 1:
                    image = Image.fromarray(canvas)
                    send_to_ai(model, image)

        # Combine the image and canvas
        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_response(request):
    global response_text
    return JsonResponse({'response': response_text})
