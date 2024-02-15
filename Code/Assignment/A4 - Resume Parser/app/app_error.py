# Import libraries
from flask import Flask, render_template, request, send_file, jsonify
from utils import readPDF
import pandas as pd
from io import StringIO

# web Flask
app = Flask(__name__)

resume_info = None  # Declare a variable to store resume information

@app.route('/', methods=['GET', 'POST'])
def index():
    global resume_info

    if request.method == 'GET':
        # return the HOME page
        return render_template('index.html', resume_info=resume_info)

    if request.method == 'POST':
        if 'resume' not in request.files:
            return 'No file part'

        # get the upload file
        resume = request.files['resume']

        # check file is included or not
        if resume.filename == '':
            return 'No selected file'

        # Call the readPDF function to extract information
        resume_info, df = readPDF(resume)
        print('File received successfully!')
        print('Resume Info:', resume_info)

        # Convert sets in resume_info to lists
        for key, value in resume_info.items():
            if isinstance(value, set):
                resume_info[key] = list(value)

        # Create a buffer for the CSV file
        buffer = StringIO()
        df.to_csv(buffer, index=False)

        # Set up the response to download the CSV file
        buffer.seek(0)
        csv_data = buffer.getvalue()

        # Prepare response data
        response_data = {
            'resume_info': resume_info,
            'csv_data': csv_data
        }
        print(response_data['resume_info'])
        print(response_data['csv_data'])

        return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
