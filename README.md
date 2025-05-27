# 🖼️ Image Augmentation Web App

A simple and powerful web app for applying both **manual** and **automatic image augmentations**, built using **Flask** and **Albumentations**.

## 🌐 Live Demo

👉 [Check it out here](https://image-augmentation-app-nit0.onrender.com/)

## 🚀 Features

- ✅ Upload any image (JPG, PNG)
- 🔧 Apply **manual augmentations** like:
  - Rotation
  - Horizontal/Vertical Flip
  - Brightness adjustment
  - Zoom/Crop
- ⚙️ Run **automatic augmentation** using simple Python logic
- ⚡ View and download the augmented images
- 🔄 Augment multiple images quickly

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: Albumentations, OpenCV

## 📂 Project Structure

```

image-augmentation-app/
├── app.py
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/
│   └── index.html
├── augmentations/
│   ├── manual.py
│   └── automatic.py
├── requirements.txt
└── README.md

````

## 🧪 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/image-augmentation-app.git
   cd image-augmentation-app
````

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:

   ```bash
   python app.py
   ```

5. Open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 💡 Future Improvements

* Add more augmentation options
* Add batch upload and download support
* Deploy the app online
* Option to compare before and after images side-by-side

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

Made with ❤️ using Flask and Python

```

---

Let me know if you'd like a version that includes deployment steps (e.g., Render, Heroku), or if you want me to help generate screenshots or GIFs.
```
