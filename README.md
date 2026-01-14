
# 📌 Project Overview

Dog breed identification is a challenging computer vision task due to high inter-class similarity and variations in pose, lighting, and background.
This project leverages MobileNetV2, a lightweight and efficient CNN architecture, to build a scalable and accurate dog breed classifier.

# Screenshots
https://github.com/Tridip-2004/Dog-s_Breed_prediction_With_Render_Deployment-/blob/main/Screenshot1.png
https://github.com/Tridip-2004/Dog-s_Breed_prediction_With_Render_Deployment-/blob/main/Screenshot2.png

# 🔹 Key Highlights

✅ 157 dog breed classes

✅ MobileNetV2 pretrained on ImageNet

✅ Transfer learning + fine-tuning

✅ Optimized for deployment

✅ Streamlit web application

# 🧠 Model Architecture

Base Model: MobileNetV2 (pretrained on ImageNet)

Input Shape: (224 × 224 × 3)

Custom Classification Head:

Global Average Pooling

Dense (ReLU)

Dropout (regularization)

Dense Softmax output (157 classes)

# 🔧 Fine-Tuning Strategy

Initial layers frozen for feature extraction

Top layers unfrozen for fine-tuning

Lower learning rate for stable convergence

# 📂 Dataset

Total Classes: 157 dog breeds

Image Type: RGB images

Preprocessing:

Resizing to 224×224

Normalization

Data augmentation (rotation, flip, zoom)

# 📌 Dataset organized using class-wise folders for training and validation.

⚙️ Technologies Used
Category	Tools
Programming	Python
Deep Learning	TensorFlow, Keras
Model	MobileNetV2
Image Processing PIL
Web App	Streamlit
Will be ready to Deploy on Render
Version Control	Git, GitHub
# 🚀 Model Training

Loss Function: Categorical Crossentropy

Optimizer: Adam

Learning Rate: 0.001

Epochs: Tuned for optimal performance

Evaluation Metric: Accuracy

The model was trained using transfer learning, followed by fine-tuning to improve performance on dog breed–specific features.

# 🖥️ Streamlit Web Application
Features:

📤 Upload dog image

🔍 Predict breed in real-time

📊 Confidence score visualization

# ⚡ Fast inference

streamlit run app.py

📁 Project Structure
├── app.py
├── dog_breed_classifier.h5
├── class_names.txt
|--- Dockerfile
├── requirements.txt
├── README.md
└── dataset/
    ├── train/
    └── validation/

# 🛠️ Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/your-username/dog-breed-classification.git
cd dog-breed-classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

# 📊 Results

✔ High classification accuracy across 157 breeds

✔ Efficient inference using MobileNetV2

✔ Robust performance on unseen images

The fine-tuned model significantly improves accuracy compared to a frozen base model.

# 🌐 Deployment

The application will be ready to deploy using Render, ensuring:

Reliable performance

Scalable infrastructure

Easy accessibility

# 🔮 Future Enhancements

🔹 Add top-3 predictions

🔹 Integrate Grad-CAM visualization

🔹 Improve accuracy with EfficientNet

🔹 Mobile app deployment

🔹 Multilingual UI support

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome!
Feel free to fork the repository and submit a pull request.

# 📜 License

This project is licensed under the MIT License.

# 👨‍💻 Author

Tridip Panja
🎓 AI / Machine Learning Enthusiast
💻 Deep Learning | Computer Vision | Deployment
