from flask import Flask, request
from flask_cors import CORS
import os
import csv
from csvHandler import process_csv

app = Flask(__name__)
CORS(app)  # This will allow CORS for all domains

@app.route('/upload', methods=['POST'])
def upload_xml():
    print("Request received")
    print(request.files)
    if 'file' not in request.files:
        print("No file found")
        return 'No file found', 400
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return 'No file selected', 400
    # Get the file extension
    file_extension = os.path.splitext(file.filename)[1]

    # Check if the file is XML or CSV
    if file_extension.lower() == '.xml':
        # Process XML file
        # Your code here
        pass
    elif file_extension.lower() == '.csv':
        # Process CSV file
        # Save the uploaded CSV file as data.csv
        file.save('new.csv')
        process_csv('new.csv')
    else:
        # Invalid file format
        print("Invalid file format")
        return 'Invalid file format', 400
    # Process the file here
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()
