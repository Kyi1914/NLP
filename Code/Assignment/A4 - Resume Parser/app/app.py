# Import libraries
from flask import Flask, render_template, send_file,request, jsonify
from utils import readPDF
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
from io import StringIO

# web Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    # return the HOME page
    if request.method == 'GET':
        return render_template ('index.html', prompt = '')
    
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
        
        if 'download' in request.form:
            # If the download button is clicked
            # Create a buffer for the CSV file
            buffer = StringIO()
            df.to_csv(buffer, index=False)

            # Set up the response to download the CSV file
            buffer.seek(0)
            return send_file(buffer,
                             as_attachment=True,
                             download_name='resume_info.csv',
                             mimetype='text/csv')
            
        # Pass the information to the template
        return render_template('index.html', resume_info=resume_info)

port_number = 8000

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = port_number)
