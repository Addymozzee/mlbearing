from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/src/components/data_transformation.py')
from src.components.data_transformation import DataTransformation, CustomException
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from shutil import rmtree


app = Flask(__name__, template_folder='your_template_folder', static_folder='your_static_folder')
application=Flask(__name__)

app=application

## Route for a home page

# @app.route('/')
# def index():
#     return render_template('index.html') 

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_bearing_data():
#     pred_df = None
#     try:
#         if request.method == 'GET':
#             return render_template('home.html')
#         else:
#             # Retrieve the files from the 'your_data_path' file input
#             your_data_path = request.files.getlist('your_data_path_file')  # Ensure the name matches with your HTML form
#             your_data_path_text = request.form['your_data_path_text']  # Ensure the name matches with your HTML form
            
#             # Path to your CustomData class and processing pipeline
#             data = CustomData(your_data_path, your_data_path_text)
#             pred_df = data.get_data_directory_path()
            
#             # Initialize the prediction pipeline; this now handles data transformation automatically
#             predict_pipeline = PredictPipeline()
#             results = predict_pipeline.predict(pred_df)  # The pipeline handles transformation now
            
#             return render_template('home.html', results=results[0])    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return render_template('home.html', results=f"An error occurred: {str(e)}") # I modified this to return the error message
#     finally:
#         if pred_df:
#             rmtree(pred_df)  # Cleanup temporary directory

# if __name__ == "__main__":
#     app.run(debug=True)

# # Define a custom exception for data processing errors
# class CustomDataProcessingError(Exception):
#     pass

# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_bearing_data():
#     pred_df = None
#     results = {
#         'SVM': None,
#         'Logistic Regression': None,
#         'Random Forest': None,
#         'XGBoost': None,
#         'LSTM': None
#     }

#     try:
#         if request.method == 'GET':
#             return render_template('home.html', results=results)
#         else:
#             # Retrieve the files from the 'your_data_path' file input
            # your_data_path = request.files.getlist('your_data_path_file')
            # your_data_path_text = request.form['your_data_path_text']

#             # Path to your CustomData class and processing pipeline
            # data = CustomData(your_data_path, your_data_path_text)
            # pred_df = data.get_data_directory_path()
       
#             # Initialize the prediction pipeline
#             predict_pipeline = PredictPipeline()

#             # Assuming the pipeline's predict method now returns a dictionary of predictions and accuracies
#             results = predict_pipeline.predict(pred_df)

#     except FileNotFoundError as e:
#         # Handle the case when the file is not found
#         return render_template('home.html', error=f"File not found: {str(e)}")

#     except CustomDataProcessingError as e:
#         # Handle a specific custom exception related to data processing
#         return render_template('home.html', results=results, error=f"Data processing error: {str(e)}")

#     except Exception as e:
#         # Handle other unexpected exceptions
#         return render_template('home.html', results=results, error=f"An error occurred: {str(e)}")

#     finally:
#         if pred_df:
#             rmtree(pred_df)  # Cleanup temporary directory

#     # Render the template with the results
#     return render_template('home.html', results=results)

# if __name__ == "__main__":
#     app.run(debug=True)
    


# List of model types
model_types = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LSTM']

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_bearing_data():
    results = {}  # Dictionary to store prediction results for each model type

    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            # Retrieve the input data here, e.g., from request.form or request.files
            your_data_path = request.files.getlist('your_data_path_file')
            your_data_path_text = request.form['your_data_path_text']

            # Retrieving the input data from the temporary directory
            data = CustomData(your_data_path, your_data_path_text)
            pred_df = data.get_data_directory_path()

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Make predictions for each model type
            predictions = predict_pipeline.predict(pred_df)

            # Store the predictions in the results dictionary with the model type as the key
            for model_type, preds in predictions.items():
                results[model_type] = {'prediction': preds}

    except Exception as e:
        # Handle exceptions or errors here
        return render_template('home.html', results=results, error=str(e))

    # Render the template with the results
    return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)



