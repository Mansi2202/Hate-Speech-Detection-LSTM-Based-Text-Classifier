Sure! Here's the markdown (.md) code for the complete README.md file you provided:

README.md (Markdown Code)

````markdown
# Hate Speech Detection using LSTM (Hate-Code)

A deep learning-based system that detects hate or abusive speech in tweets using an LSTM (Long Short-Term Memory) network trained on a combination of imbalanced and raw Twitter datasets.

Built with TensorFlow, Keras, and Streamlit for a clean UI to test your own inputs.

## 🚀 Features

- Cleans and preprocesses tweet data (removes noise, stopwords, punctuation, etc.)
- Combines two datasets for robust hate speech classification
- Uses LSTM architecture with word embeddings for training
- Saves trained model and tokenizer for future predictions
- Streamlit app for interactive testing with custom inputs

## 🧠 Model Details

- Architecture: Embedding Layer → SpatialDropout → LSTM → Dense
- Tokenizer: Keras Tokenizer
- Loss: Binary Crossentropy
- Optimizer: RMSprop
- Accuracy evaluated on test set using confusion matrix

## 📁 Project Structure

project/  
├── Hate_code.py              # Model training and saving  
├── app.py                    # Streamlit-based inference app  
├── tokenizer.pickle          # Saved tokenizer  
├── model.h5                  # Saved LSTM model  
├── data/  
│   ├── imbalanced_data.csv   # Dataset with binary hate labels  
│   └── raw_data.csv          # Dataset with multi-class offensive labels  
├── requirements.txt          # All required Python libraries  
└── README.md                 # Project documentation

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/hate-speech-lstm.git
cd hate-speech-lstm
````

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

## 💻 Usage

1. First, train and save the model:

```bash
python Hate_code.py
```

This will generate model.h5 and tokenizer.pickle.

2. Then run the Streamlit app:

```bash
streamlit run app.py
```

3. Enter any custom sentence in the input box and hit “Submit” to detect whether the text is hateful or not.

## ✅ Sample Input/Output

Input:

```
You people are disgusting and should be banned.
```

Output:

```
Prediction: Hate and Abusive
```

---

Input:

```
Hope you all are doing great today!
```

Output:

```
Prediction: No Hate
```

## 🧠 Technologies Used

* Python
* Pandas, Numpy
* NLTK (for text preprocessing)
* TensorFlow / Keras (for LSTM)
* Matplotlib, Seaborn (EDA)
* Streamlit (for deployment)

## 🏁 Future Work

* Add support for batch predictions from CSVs
* Integrate explainability (e.g., SHAP values)
* Deploy on cloud (e.g., Hugging Face Spaces or Streamlit Cloud)

## 🙌 Acknowledgements

* Twitter Hate Speech Datasets from Kaggle
* NLTK for robust NLP preprocessing
* TensorFlow and Keras for deep learning framework
* Streamlit for fast and interactive UI

---

```

