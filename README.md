# 🔮 NextWord AI

A **Deep Learning web application** that predicts the most likely next word based on the text entered by the user.
The application uses a trained **Long Short-Term Memory (LSTM)** neural network  to understand the input text and generate an intelligent next-word prediction.

---

## 🚀 Features

- Interactive **Streamlit UI** for easy text input.
- Enter a sentence or phrase for prediction.
- Prediction using a trained **LSTM model**.
- Uses the most recent words from the input sequence.
- Displays the predicted **next word** instantly.

---

## 📂 Project Structure

```text
├── lstm_model (1).h5       # Trained LSTM model
├── tokenizer.pkl            # Trained text tokenizer
├── max_len.pkl              # Maximum sequence length used during training
├── app.py                   # Streamlit application
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🎯 Usage

1. Open the application here: **[Run the App](https://nextword-ai-app.streamlit.app/)**
2. Enter a sentence or phrase in the text box.
3. Click **Predict next word**.
4. View the predicted next word instantly.

---

## 🧠 Model Details

* **Model:** Long Short-Term Memory (LSTM)
* **Prediction:** Next-word probability prediction
* **Output:** Most probable next word

The model produces a probability distribution over the vocabulary, and the word with the highest predicted probability is selected as the next-word prediction.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit
* Pickle

---

## ⚠️ Disclaimer

This application is a prototype created for **educational and demonstration purposes only**.
The underlying model has been trained on a relatively small dataset, so its predictions may not always be accurate, relevant, or grammatically correct.
The application is intended to demonstrate the concept of predictive text rather than serve as a production-ready typing assistant.
Predictions should therefore be treated as suggestions, and incorrect or unexpected results may occur.

---

## 📸 Screenshot

<img width="1761" height="827" alt="Screenshot 2026-08-21 144702" src="https://github.com/user-attachments/assets/587590c5-2ccd-4e1c-aa35-9f59cac52c4b" />


---

## 🤝 Contribution

Pull requests are welcome. For any changes, please open an issue first to discuss what you would like to change.
