# Import libraries
from flask import Flask, render_template, send_file, request, jsonify
from utils import readPDF
import pandas as pd
from io import StringIO

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', prompt='')

    if request.method == 'POST':
        if 'resume' not in request.files:
            return 'No file part'

        resume = request.files['resume']

        if resume.filename == '':
            return 'No selected file'

        resume_info, csv_data = readPDF(resume)
        print('File received successfully!')
        print('Resume Info:', resume_info)

        # Pass the information and CSV data to the template
        return render_template('index.html', resume_info=resume_info, csv_data=csv_data)

    if 'download' in request.form:
        csv_data = request.form['csv_data']

        if csv_data:
            buffer = StringIO(csv_data)
            return send_file(buffer,
                             as_attachment=True,
                             download_name='resume_info.csv',
                             mimetype='text/csv')

    return 'Invalid request'

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)
