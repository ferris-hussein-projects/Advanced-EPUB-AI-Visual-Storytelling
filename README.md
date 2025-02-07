# **Advanced EPUB AI: Visual Storytelling**

## **Overview**
This project was inspired by my wife, who often showed me amazing fan art of her favorite books on TikTok. Seeing how much joy she found in those visual interpretations, I wanted to create a tool that could **automate the process of bringing book characters and scenes to life** using AI-generated images.  

**Advanced EPUB AI: Visual Storytelling** is an **interactive, web-based application** that allows users to:
- **Upload an EPUB book**
- **Extract character descriptions** using NLP**
- **Summarize the most prominent characters**
- **Generate AI-based character portraits and scene illustrations**
- **Reconstruct the EPUB** by inserting the generated images
- **Convert the final EPUB to MOBI, AWZ3 and PDF format for compression and compatibility with various e-readers**  

This project enhances **traditional reading experiences** by making books **more immersive and visually engaging**.

---

## **Key Features**

### 📖 EPUB Processing & Character Mention Extraction  
- Extracts **character mentions** and **scene details** using **spaCy NLP**
- Identifies **top N most relevant characters** for visualization  
- **Uses a pretrained Transformer-based NLP model (`en_core_web_trf`)** to extract **PERSON entities** from text  

### **Character Summarization & Correction**
- Aggregates **textual descriptions** of characters from the book  
- Generates **concise AI-driven summaries**
- Ensures accuracy in **physical traits (hair color, eye color, skin tone, personality, etc.)**  
- **Interactive Editing:** Users can **edit and refine** AI-generated summaries directly in the app before finalizing character portraits  
- Corrects inconsistencies by **merging duplicate names** (e.g., "Jon" vs. "Jonathan")  

### AI-Generated Portraits & Scenes  
- Uses **DALL·E** to create **highly detailed, fantasy-style portraits**
- Generates **scenic images** for key moments in the book
- Ensures AI-generated visuals match the story’s **tone and descriptions**
- **Note:** Some images may fail to generate if OpenAI’s API flags content as a **violation**. In such cases:
  - Try **regenerating the images**.
  - **Modify the character or scene descriptions** in the respective JSON files before retrying.
  - **Refine prompts** to avoid restricted content.

### **EPUB Reconstruction, Customization & MOBI Conversion**
- **Inserts character portraits** at the start of the book  
- **Places scene illustrations** immediately after their respective chapters  
- **Allows users to select a character portrait as the book cover**  
- **Saves and converts the final EPUB to MOBI format** for:
  - **Better compression**
  - **Compatibility with Kindle and other e-readers**
  - **Reduced file size for easier sharing and storage**  

### **Streamlit-Based Interactive Web UI**
- **No complex setup needed**—just open in a browser and use a simple UI  
- **Step-by-step guidance** to streamline the process from upload to EPUB reconstruction  
- Users can **modify AI-generated summaries**, **select key characters**, and **review generated images**  
- Provides an **image gallery** for reviewing character portraits and scene illustrations before EPUB finalization  

### **Custom API Key Input & Model Configuration**
- Users can **enter their own OpenAI API keys** for model access  
- The app **supports modifying model variables**, allowing users to select different AI models  
- Models used in this project:
  - **GPT-4o** (used for character summarization)
  - **DALL·E 3** (used for character portraits and scene images)
- **Note:** Different models have different pricing structures—**API charges apply based on OpenAI's pricing model.**  
- ⚠ **Handling API Denials:**  
  - If OpenAI **flags content as a violation**, the image **will not be generated**.  
  - If this happens, users can:
    - Try **regenerating** the images.
    - Modify **character descriptions** in `Character_Summaries.json`.
    - Modify **chapter summaries** in `Chapter_Scenes.json` to remove sensitive content.

---

## **Understanding the JSON Files**
The application stores extracted data in JSON files for easy reference and editing:

- **`Character_Profiles.json`**  
  Stores raw extracted text from the book related to each character before summarization.

- **`Character_Summaries.json`**  
  Contains AI-generated summaries for the top characters, which are used to generate AI portraits.

- **`Chapter_Scenes.json`**  
  Stores AI-generated **scene summaries** based on chapter content. This is used to generate scenic images.

- **How to Modify JSONs:**  
  - If an **AI-generated image fails**, try **editing the corresponding JSON file** before regenerating images.  
  - The app will use updated JSON data when running the next **EPUB reconstruction**.  

---

## **Character Mention Extraction with spaCy**
One of the most important steps in the **character processing pipeline** is **identifying and extracting character mentions** from the EPUB text.

### **How It Works**
1. **Load the NLP Model**  
   - The application uses **spaCy’s `en_core_web_trf`**, a transformer-based NLP model trained on large amounts of textual data.  
   - This model is specifically designed to **identify named entities** such as **people, locations, and organizations**.
   ```python
   import spacy
   nlp = spacy.load("en_core_web_trf")
   ```

2. **Extract Mentions of Character Names**
   - The tool processes each chapter of the EPUB using spaCy and **extracts "PERSON" entities** from the text.
   ```python
   doc = nlp(chapter_text)
   character_mentions = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
   ```

3. **Track Character Mentions**
   - Each detected **character name** is **stored and counted**, helping determine which characters are the most important.
   - The tool also **groups similar character names together** (e.g., **"Jon" and "Jonathan"**) to ensure accuracy.
   ```python
   from collections import Counter
   character_counts = Counter(character_mentions)
   ```

4. **Refinement & Cleaning**
   - The application **merges alternate spellings or abbreviations** of names using **fuzzy matching**.
   - This step ensures that **different references to the same character are counted as one**.

---

## **Installation & Setup**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
streamlit run epub_ai.py
```

---

## **Future Improvements**
- **Expand model support** for different AI services  
- **Improve scene descriptions** with richer detail  
- **Custom model training** for **better character summarization**  

---

### **Final Notes**
- The app allows **custom API key entry** and **modification of model parameters**.  
- **EPUBs are converted to MOBI format** for **smaller file size and compatibility with e-readers**.  
- Users can **add custom images** if they follow the correct naming conventions.  
- **API content violations** may prevent image generation. In such cases:
  - **Modify character or scene summaries** in JSON files.
  - **Refine the prompts** to better fit OpenAI’s policies.
  - **Retry the image generation process**.
