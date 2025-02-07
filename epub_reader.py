import os
import json
import time
import requests
from collections import Counter
from ebooklib import epub
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import spacy
import openai
import streamlit as st

###############################################################################
# Streamlit App Setup
###############################################################################
st.set_page_config(page_title="Advanced EPUB AI: Visual Storytelling", layout="wide")
st.title("Advanced EPUB AI: Visual Storytelling")
st.write(
    "**Upload an EPUB, extract & summarize characters (choose how many), optionally edit those summaries, " 
    "then generate AI-based portrait and scene images. Finally, reconstruct the EPUB with generated images.**"
)

###############################################################################
# Sidebar for API Keys and Input Widgets
###############################################################################
with st.sidebar:
    st.header("OpenAI API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    openai_org = st.text_input("OpenAI Organization ID (Optional)", type="password")
    top_n = st.selectbox(
        "Select the number of top characters to summarize",
        [1,2,4,8],
        key="top_n_selectbox"
    )
    epub_file = st.file_uploader("Upload an EPUB file", type="epub")

if openai_api_key:
    openai.api_key = openai_api_key
    print("OpenAI API Key Set.")
if openai_org:
    openai.organization = openai_org
    print("OpenAI Organization ID Set.")

###############################################################################
# Global / Session State Structures
###############################################################################
if "global_character_profiles" not in st.session_state:
    st.session_state["global_character_profiles"] = {}
if "character_mentions" not in st.session_state:
    st.session_state["character_mentions"] = Counter()
if "character_mentions_per_chapter" not in st.session_state:
    st.session_state["character_mentions_per_chapter"] = {}
if "chapter_scenes" not in st.session_state:
    st.session_state["chapter_scenes"] = {}
if "top_n_summaries" not in st.session_state:
    st.session_state["top_n_summaries"] = {}
if "book_title" not in st.session_state:
    st.session_state["book_title"] = "Throne_Of_Glass"
if "epub_processed" not in st.session_state:
    st.session_state["epub_processed"] = False
if "uploaded_epub_path" not in st.session_state:
    st.session_state["uploaded_epub_path"] = None
if "images_generated" not in st.session_state:
    st.session_state["images_generated"] = False

###############################################################################
# Utility Functions
###############################################################################
def sanitize_name(name):
    """Convert a name into a file-friendly format."""
    return "_".join(part.capitalize() for part in name.split())

def extract_first_name(name):
    """Extract the first token from a name string."""
    return name.split()[0]

def save_image(image_url, path):
    """Download and save an image from a URL."""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(path, 'wb') as file:
                file.write(response.content)
            print(f"Image saved: {path}")
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

def merge_similar_names(characters):
    """
    Merge characters that are very similar in name (e.g., "Dany" vs "Daenerys")
    to reduce duplicates.
    """
    merged = {}
    threshold = 85
    for full_name, sentences in characters.items():
        first_name = extract_first_name(full_name)
        found_match = False
        for key in list(merged):
            if fuzz.token_sort_ratio(sanitize_name(full_name), sanitize_name(key)) > threshold or \
               fuzz.partial_ratio(first_name, key) > threshold:
                merged[key].update(sentences)
                found_match = True
                break
        if not found_match:
            merged[sanitize_name(full_name)] = sentences
    return merged

###############################################################################
# Character Summaries Generation
###############################################################################
def generate_top_n_character_summaries(profile_path, top_n=5):
    """
    1) Load raw character profiles from JSON.
    2) Identify the top N characters by total text length.
    3) Generate short summaries (20 sentences) using OpenAI.
    4) Return a dictionary: {character_name: short_summary}.
    """
    try:
        with open(profile_path, "r", encoding="utf-8") as file:
            all_profiles = json.load(file)
    except Exception as e:
        print(f"Error loading Character_Profiles.json: {e}")
        return {}

    sorted_characters = sorted(
        all_profiles.keys(),
        key=lambda c: sum(len(segment) for segment in all_profiles[c]),
        reverse=True
    )
    top_n_chars = sorted_characters[:top_n]
    character_summaries = {}

    for character in top_n_chars:
        print(f"Generating summary for character: '{character}'...")
        if character not in all_profiles:
            print(f" - No raw profile found for {character}, skipping summary.")
            character_summaries[character] = "Summary unavailable."
            continue

        full_description = " ".join(all_profiles[character])
        prompt = f"""
{full_description}

Carefully analyze the provided text and create an accurate character description in exactly 20 sentences.
Prioritize features that are mentioned most frequently, ensuring the description aligns with the original text.
Focus on specific physical characteristics such as hair color, facial features, clothing style, posture, age, and 
overall demeanor, while also capturing the character's personality, presence, and role in the story.
Use vivid, sensory-rich language to bring the character to life, emphasizing their most defining traits.
Avoid speculation and rely only on details explicitly mentioned in the text, prioritizing repeated or
emphasized elements. Maintain a natural, immersive tone that matches the style of the story. Confirm
the hair color AND skin color AND eye color and age range by referring to the LLM's knowledge of the character, 
ensuring consistency with the character's established traits in any relevant media or texts.
"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[  
                    {"role": "system", "content": "You are a literary character summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=1000
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary for {character}: {e}")
            summary = "Summary unavailable."
 
        character_summaries[character] = summary

    return character_summaries

###############################################################################
# Portrait Generation
###############################################################################
def generate_portrait_images_from_summaries(character_summaries, output_folder, regenerate=False):
    """
    Generate portrait images for the given character summaries.
    The portrait image is saved as <Character>_1.png.
    """
    print("Generating portrait images for top characters...")
    summaries_path = os.path.join(output_folder, "Character_Summaries.json")
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(character_summaries, f, indent=4)

    character_image_folder = os.path.join(output_folder, "Character_Images")
    os.makedirs(character_image_folder, exist_ok=True)

    for character, summary in character_summaries.items():
        print(f"Generating portrait image for character: '{character}'...")
        portrait_prompt = (
            f"Create a highly detailed, cinematic-style portrait of {character}. "
            f"Ensure absolute accuracy in depicting their hair color, skin tone, age, and eye color as described: {summary}. "
            "The portrait should be fantasy-themed, very realistic with high detail and dramatic lighting. "
            "Do NOT include any text in the image. The image should just have the character with no context."
        )

        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=portrait_prompt,
                n=1,
                quality="hd",
                size="1024x1024"
            )
            image_urls = response.data
            image_filename = f"{sanitize_name(character)}_1.png"
            image_path = os.path.join(character_image_folder, image_filename)
            save_image(image_urls[0].url, image_path)
        except Exception as e:
            print(f"Error generating portrait for {character}: {e}")

    print("Character portrait image generation complete!")

###############################################################################
# Scene/Chapter Summaries and Images
###############################################################################
def generate_scene_summary(text):
    """Generate a 10-sentence summary from the chapter text."""
    prompt = f"""
{text}

Carefully analyze the provided chapter text and produce an accurate summary of its key events in exactly 10 sentences.
Prioritize events that are mentioned most frequently and are central to the chapter's progression, ensuring the summary aligns with the original narrative.
Focus on specific moments such as plot twists, conflicts, resolutions, setting changes, and character decisions that drive the story forward.
Use vivid, sensory-rich language to capture the mood and pace of the events, emphasizing their impact on the overall narrative.
Avoid speculation and rely only on details explicitly mentioned in the text, prioritizing repeated or emphasized elements.
Maintain a natural, immersive tone that matches the style of the chapter while ensuring clarity and coherence.
Ensure that each sentence contributes meaningfully to the overall understanding of the chapter’s events.
Confirm the chronological order and causality of events by closely referring to the text's narrative structure.
Double-check that the summary is faithful to the original details without adding extraneous interpretations.
Finally, verify that the final output contains exactly 10 well-crafted sentences that provide a comprehensive overview of the chapter.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[  
                {"role": "system", "content": "You are a literary summarizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating scene summary: {e}")
        return "Scene summary unavailable."

def generate_scene_images(scene_summary, output_folder, chapter_key):
    """
    Generate exactly 2 scenic images for the given scene summary.
    """
    os.makedirs(output_folder, exist_ok=True)
    scene_prompt = f"""
Carefully analyze the provided scene summary: {scene_summary}. Craft a detailed, fantasy-themed description of the environment, 
emphasizing high detail and dramatic lighting. Focus exclusively on the scenery—highlight natural landscapes, mystical elements, and atmospheric 
conditions—without introducing any characters or text. Rely solely on the details present in the summary, using vivid, sensory-rich language to 
bring the environment to life. Ensure the description aligns with the established aesthetic of fantasy settings as seen in relevant media. 
No characters present. No text in the image, only environment. The image should just have the scenery with no context.
    """

    for i in range(1):
        try:
            response = openai.images.generate(
                model="dall-e-3",
                prompt=scene_prompt,
                n=1,
                quality="hd",
                size="1024x1024"
            )
            image_url = response.data[0].url
            image_path = os.path.join(output_folder, f"{chapter_key}_{i+1}.png")
            save_image(image_url, image_path)
        except Exception as e:
            print(f"Error generating scene image for {chapter_key}_{i}: {e}")

def save_data_and_generate_scene_images():
    """
    Save final chapter scenes data and generate scenic images for each chapter.
    Scene images are saved in a folder named "Chapter_Images".
    """
    base_directory = sanitize_name(st.session_state["book_title"])
    scene_data_path = os.path.join(base_directory, "Chapter_Scenes.json")
    with open(scene_data_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state["chapter_scenes"], f, indent=4)

    chapter_image_folder = os.path.join(base_directory, "Chapter_Images")
    for chapter_key, chapter_data in st.session_state["chapter_scenes"].items():
        generate_scene_images(chapter_data["summary"], chapter_image_folder, chapter_key)
    print("Chapter scene images generated!")

###############################################################################
# EPUB Processing Functions
###############################################################################
def extract_character_descriptions(book):
    """
    Extract PERSON entities from each chapter using spaCy.
    Updates session_state with character mentions and profiles.
    """
    print("Loading spaCy model (en_core_web_trf)...")
    nlp = spacy.load("en_core_web_trf")
    all_characters = {}
    chapter_num = 1

    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            text = BeautifulSoup(item.get_content(), 'html.parser').get_text()
            doc = nlp(text)
            chapter_key = f"Chapter_{chapter_num}"
            print(chapter_key)
            st.session_state["character_mentions_per_chapter"][chapter_key] = []

            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    character = ent.text.strip()
                    sentence = ent.sent.text.strip()
                    st.session_state["character_mentions"][character] += 1
                    all_characters.setdefault(character, set()).add(sentence)
                    st.session_state["character_mentions_per_chapter"][chapter_key].append(character)

            chapter_num += 1

    merged_characters = merge_similar_names(all_characters)
    st.session_state["global_character_profiles"].update(merged_characters)
    print("Character extraction complete!")
    return st.session_state["character_mentions_per_chapter"]

def extract_scene_summaries(book, max_chapters=1):
    """
    Summarize each chapter (up to max_chapters) in 10 sentences.
    """
    chapter_scenes = {}
    chapter_num = 1
    for item in book.get_items():
        if chapter_num > max_chapters:
            break
        if isinstance(item, epub.EpubHtml):
            text = BeautifulSoup(item.get_content(), 'html.parser').get_text()
            scene_summary = generate_scene_summary(text)
            chapter_key = f"Chapter_{chapter_num}"
            print(chapter_key)
            chapter_scenes[chapter_key] = {"summary": scene_summary}
            print(f"Processed {chapter_key} -> Summary: {scene_summary[:60]}...")
            chapter_num += 1
    return chapter_scenes

def process_epub_for_extraction(file_path, top_n):
    """
    Process the EPUB file:
      1) Read EPUB metadata.
      2) Extract characters and scenes.
      3) Save raw character profiles.
      4) Generate top-n character summaries.
    """
    print(f"Reading EPUB file: {file_path}")
    book = epub.read_epub(file_path)
    metadata_title = book.get_metadata('DC', 'title')
    st.session_state["book_title"] = metadata_title[0][0] if metadata_title else "Unknown_Book"

    extract_character_descriptions(book)
    chapter_scenes = extract_scene_summaries(book, max_chapters=1)
    st.session_state["chapter_scenes"] = chapter_scenes

    base_directory = sanitize_name(st.session_state["book_title"])
    os.makedirs(base_directory, exist_ok=True)
    profile_path = os.path.join(base_directory, "Character_Profiles.json")
    profiles_dict = {k: list(v) for k, v in st.session_state["global_character_profiles"].items()}
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profiles_dict, f, indent=4)
        
    st.session_state["top_n_summaries"] = generate_top_n_character_summaries(profile_path, top_n)
    st.session_state["epub_processed"] = True
    print("EPUB processing complete!")
    return True

###############################################################################
# Session State Initialization & Helper for EPUB Processing
###############################################################################
def initialize_session_state():
    if "book_title" not in st.session_state:
        st.session_state["book_title"] = "Throne_Of_Glass"
    if "epub_processed" not in st.session_state:
        st.session_state["epub_processed"] = False
    if "top_n_summaries" not in st.session_state:
        st.session_state["top_n_summaries"] = {}

def process_epub_file(epub_file, top_n, base_directory, image_folder):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, epub_file.name)
    
    try:
        with open(file_path, "wb") as f:
            f.write(epub_file.getbuffer())
        st.success(f"File uploaded successfully: {epub_file.name}")
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return False
    
    # Store the uploaded EPUB path in session state for later reconstruction.
    st.session_state["uploaded_epub_path"] = file_path

    if not st.session_state["epub_processed"]:
        process_epub_for_extraction(file_path, top_n)
        st.session_state["epub_processed"] = True
    else:
        st.success("EPUB already processed!")
    
    st.success("EPUB file processed successfully!")
    return True

###############################################################################
# UI Functions for Confirming Character Summaries
###############################################################################
def display_character_summaries(top_n):
    with st.container():
        st.subheader(f"Top-{top_n} Character Summaries (Edit if Needed)")
        for character, summary in st.session_state["top_n_summaries"].items():
            new_summary = st.text_area(
                label=f"Summary for {character}",
                value=summary,
                key=f"summary_{character}",
                height=200
            )
            st.session_state["top_n_summaries"][character] = new_summary

def confirm_character_summaries(base_directory):
    with st.container():
        if st.button("Confirm Summaries and Generate Images"):
            # Generate character portrait images
            generate_portrait_images_from_summaries(st.session_state["top_n_summaries"], base_directory, regenerate=False)
            st.success("Character portrait images generated!")
            # Generate scene images from chapter summaries
            save_data_and_generate_scene_images()
            st.success("Scene images generated and saved!")
            # Set a flag so that reconstruction and image gallery are now available.
            st.session_state["images_generated"] = True

###############################################################################
# EPUB Reconstruction Function
###############################################################################

def reconstruct_epub():
    import re
    import subprocess

    # Ensure that we have an uploaded EPUB file to work from.
    uploaded_path = st.session_state.get("uploaded_epub_path")
    if not uploaded_path:
        st.error("No EPUB file path found for reconstruction.")
        return

    # Read the original EPUB using ebooklib.
    book = epub.read_epub(uploaded_path)
    base_directory = sanitize_name(st.session_state["book_title"])
    chapter_images_dir = os.path.join(base_directory, "Chapter_Images")
    character_images_dir = os.path.join(base_directory, "Character_Images")

    # ---------------------------------------------------------
    # First, gather the "normal" text chapters in the order they appear.
    # (We assume these are the EpubHtml items that are not our generated chapters.)
    # ---------------------------------------------------------
    normal_chapters = []
    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            # Exclude any chapter we might have already created for scenes or characters.
            if (not item.file_name.endswith("character_images.xhtml") and
                not item.file_name.endswith("_scene_images.xhtml")):
                normal_chapters.append(item)

    # ---------------------------------------------------------
    # Create a mapping from chapter index to a scene images chapter (if any)
    # ---------------------------------------------------------
    scene_chapters = {}  # key: chapter index (1-based), value: EpubHtml item for scene images
    for index, chapter in enumerate(normal_chapters, start=1):
        # Create a regex pattern for the chapter key.
        # The pattern ensures that after "Chapter_{index}" there is not another digit.
        pattern = re.compile(rf"^Chapter_{index}(?!\d).*\.png$", re.IGNORECASE)
        scene_files = []
        if os.path.exists(chapter_images_dir):
            for filename in sorted(os.listdir(chapter_images_dir)):
                if pattern.match(filename):
                    scene_files.append(filename)

        # If we have scene images for this chapter, create a new scene chapter.
        if scene_files:
            scene_chapter = epub.EpubHtml(
                file_name=f"Chapter_{index}_scene_images.xhtml",
                title=f"- Chapter {index} Images",
                lang="en"
            )
            # Build a basic HTML template using BeautifulSoup.
            scene_soup = BeautifulSoup("<html><head></head><body></body></html>", 'html.parser')

            # Create a container div.
            images_div = scene_soup.new_tag("div", **{"class": "generated-images"})
            scene_soup.body.append(images_div)
            
            for filename in scene_files:
                # Create an <img> tag with a relative src.
                image_tag = scene_soup.new_tag(
                    "img",
                    src=f"{base_directory}/Chapter_Images/{filename}",
                    style="max-width:100%; margin-bottom: 0.5em;"
                )
                images_div.append(image_tag)
                
                # Also add the image file as an EPUB item (if not already added).
                if not book.get_item_with_id(filename):
                    image_path = os.path.join(chapter_images_dir, filename)
                    try:
                        with open(image_path, "rb") as img_file:
                            image_content = img_file.read()
                        img_item = epub.EpubItem(
                            uid=filename,
                            file_name=f"{base_directory}/Chapter_Images/{filename}",
                            media_type="image/png",
                            content=image_content
                        )
                        book.add_item(img_item)
                    except Exception as e:
                        print(f"Error adding scene image {filename} to EPUB: {e}")

            # Set the content of the scene chapter.
            scene_chapter.set_content(str(scene_soup).encode('utf-8'))
            scene_chapters[index] = scene_chapter
            # Add the scene chapter to the book.
            book.add_item(scene_chapter)

    # ---------------------------------------------------------
    # Now, create the character images chapter (cover chapter).
    # ---------------------------------------------------------
    char_chapter = None
    if os.path.exists(character_images_dir):
        char_chapter = epub.EpubHtml(file_name="character_images.xhtml", lang="en")
        char_soup = BeautifulSoup("<html><head></head><body></body></html>", 'html.parser')
        char_div = char_soup.new_tag("div", **{"class": "character-images"})
    
        def format_character_name(filename):
            base = filename.rsplit('.', 1)[0]  # remove extension
            return base.replace("_", " ").replace("1", "").title()

        image_files = sorted([f for f in os.listdir(character_images_dir) if f.lower().endswith('.png')])
        selected_character = st.session_state.get("selected_character")
        
        if selected_character and selected_character in image_files:
            # Add the selected character image first.
            image_tag = char_soup.new_tag("img",
                                          src=f"Character_Images/{selected_character}",
                                          style="max-width:100%; margin-bottom: 0.2em;")
            char_div.append(image_tag)
            caption = char_soup.new_tag("p", 
                                        style="font-size:0.6em; text-align:center; margin-bottom:0.2em; font-style: italic; font-family: 'Trattatello', fantasy;")
            caption.string = format_character_name(selected_character)
            char_div.append(caption)
            # Also add the image file as an EPUB item if not already added.
            if not book.get_item_with_id(selected_character):
                image_path = os.path.join(character_images_dir, selected_character)
                try:
                    with open(image_path, "rb") as img_file:
                        image_content = img_file.read()
                    img_item = epub.EpubItem(
                        uid=selected_character,
                        file_name=f"Character_Images/{selected_character}",
                        media_type="image/png",
                        content=image_content
                    )
                    book.add_item(img_item)
                except Exception as e:
                    print(f"Error adding character image {selected_character} to EPUB: {e}")
        else:
            st.warning("No character image was selected; the cover will be empty.")
    
        # Add the remainder of the images ordered by popularity.
        def get_popularity(filename):
            base = filename.rsplit('.', 1)[0]
            parts = base.split('_')
            char_name = parts[0] if parts else base
            return st.session_state.get("character_mentions", Counter()).get(char_name, 0)
        
        remainder_images = [f for f in image_files if f != selected_character]
        remainder_images.sort(key=get_popularity, reverse=True)
    
        for filename in remainder_images:
            image_tag = char_soup.new_tag("img",
                                          src=f"Character_Images/{filename}",
                                          style="max-width:100%; margin-bottom: 0.2em;")
            char_div.append(image_tag)
            caption = char_soup.new_tag("p", 
                                        style="font-size:0.6em; text-align:center; margin-bottom:2em; font-style: italic; font-family: 'Trattatello', fantasy;")
            caption.string = format_character_name(filename)
            char_div.append(caption)
            if not book.get_item_with_id(filename):
                image_path = os.path.join(character_images_dir, filename)
                try:
                    with open(image_path, "rb") as img_file:
                        image_content = img_file.read()
                    img_item = epub.EpubItem(
                        uid=filename,
                        file_name=f"Character_Images/{filename}",
                        media_type="image/png",
                        content=image_content
                    )
                    book.add_item(img_item)
                except Exception as e:
                    print(f"Error adding character image {filename} to EPUB: {e}")
    
        char_soup.body.append(char_div)
        char_chapter.set_content(str(char_soup).encode('utf-8'))
        book.add_item(char_chapter)
    
    # ---------------------------------------------------------
    # Build the new spine and TOC so that the character images chapter comes first.
    # ---------------------------------------------------------
    new_spine = []
    new_toc = []
    
    if char_chapter:
        new_spine.append(char_chapter)
        toc_link = epub.Link(char_chapter.file_name, char_chapter.title or "Character Images", char_chapter.file_name)
        new_toc.append(toc_link)
    
    for idx, chapter in enumerate(normal_chapters, start=1):
        new_spine.append(chapter)
        toc_link = epub.Link(chapter.file_name, chapter.title or f"Chapter {idx}", chapter.file_name)
        new_toc.append(toc_link)
        if idx in scene_chapters:
            scene_chapter = scene_chapters[idx]
            new_spine.append(scene_chapter)
            toc_link = epub.Link(scene_chapter.file_name, scene_chapter.title or f"Scene Images for Chapter {idx}", scene_chapter.file_name)
            new_toc.append(toc_link)
    
   
    # Locate the selected character image
    selected_character = st.session_state.get("selected_character")
    character_images_dir = os.path.join(base_directory, "Character_Images")
    if selected_character:
        cover_image_path = os.path.join(character_images_dir, selected_character)
        if os.path.exists(cover_image_path):
            with open(cover_image_path, "rb") as f:
                cover_content = f.read()
            # Set the cover using EbookLib's set_cover() method.
            # The first argument is the filename (this will be used in the manifest)
            book.set_cover(selected_character, cover_content)
            st.info(f"Cover image set to {selected_character}")
        else:
            st.warning(f"Selected cover image not found: {cover_image_path}")
    else:
        st.warning("No character selected for the cover image.")

    book.spine = new_spine
    book.toc = new_toc


    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    output_epub = os.path.join(output_dir, f"{base_directory}_reconstructed.epub")
    try:
        epub.write_epub(output_epub, book)
        st.success(f"Reconstructed EPUB created: {output_epub}")
    except Exception as e:
        st.error(f"Error writing reconstructed EPUB: {e}")

    # ---------------------------------------------------------
    # Convert the EPUB to MOBI using Calibre's ebook-convert tool.
    # ---------------------------------------------------------
    output_mobi = os.path.join(output_dir, f"{base_directory}_reconstructed.mobi")
    try:
        result = subprocess.run(
            ["ebook-convert", output_epub, output_mobi],
            capture_output=True,
            text=True,
            check=True
        )
        st.success(f"MOBI file created: {output_mobi}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting EPUB to MOBI: {e.stderr}")

    # ---------------------------------------------------------
    # Convert the EPUB to AZW3 using Calibre's ebook-convert tool
    # ---------------------------------------------------------
    output_azw3 = os.path.join(output_dir, f"{base_directory}_reconstructed.azw3")
    try:
        result = subprocess.run(
            ["ebook-convert", output_epub, output_azw3],
            capture_output=True,
            text=True,
            check=True
        )
        st.success(f"AZW3 file created: {output_azw3}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting EPUB to AZW3: {e.stderr}")

    # Convert EPUB to PDF
    output_pdf = os.path.join(output_dir, f"{base_directory}_reconstructed.pdf")
    try:
        subprocess.run(
            ["ebook-convert", output_epub, output_pdf],
            capture_output=True,
            text=True,
            check=True
        )
        st.success(f"PDF file created: {output_pdf}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting EPUB to PDF: {e.stderr}")

###############################################################################
# Images Gallery Page
###############################################################################
def display_generated_images_page():
    st.header("Generated Images Gallery")
    tabs = st.tabs(["Character Images", "Chapter Images"])
    
    # Character Images Tab
    with tabs[0]:
        st.subheader("Character Images")
        char_image_folder = os.path.join(sanitize_name(st.session_state["book_title"]), "Character_Images")
        if os.path.exists(char_image_folder):
            image_files = [os.path.join(char_image_folder, f) for f in os.listdir(char_image_folder) if f.lower().endswith('.png')]
            # Group images by character name (assumes filenames like <Character>_1.png)
            char_dict = {}
            for file in image_files:
                filename = os.path.basename(file)
                if "_" in filename:
                    char_name = filename.rsplit('_', 1)[0]
                else:
                    char_name = filename
                char_dict.setdefault(char_name, []).append(file)
            for char, files in char_dict.items():
                st.markdown(f"### {char}")
                cols = st.columns(3)
                for i, file in enumerate(files):
                    cols[i % 3].image(file)
        else:
            st.info("No character images found.")
    
    # Chapter Images Tab
    with tabs[1]:
        st.subheader("Chapter Images")
        chapter_image_folder = os.path.join(sanitize_name(st.session_state["book_title"]), "Chapter_Images")
        if os.path.exists(chapter_image_folder):
            image_files = [os.path.join(chapter_image_folder, f) for f in os.listdir(chapter_image_folder) if f.lower().endswith('.png')]
            # Group images by chapter (assumes filenames like Chapter_1_1.png)
            chapter_dict = {}
            for file in image_files:
                filename = os.path.basename(file)
                parts = filename.split('_')
                if len(parts) >= 2:
                    chapter_key = "_".join(parts[:2])  # e.g., "Chapter_1"
                else:
                    chapter_key = filename
                chapter_dict.setdefault(chapter_key, []).append(file)
            for chapter, files in sorted(chapter_dict.items(), key=lambda x: int(x[0].split('_')[1])):
                st.markdown(f"### {chapter}")
                cols = st.columns(3)
                for i, file in enumerate(files):
                    cols[i % 3].image(file)
        else:
            st.info("No chapter images found.")

###############################################################################
# Main Application Function
###############################################################################
def main():
    initialize_session_state()
    base_directory = sanitize_name(st.session_state["book_title"])
    image_folder = os.path.join(base_directory, "Character_Images")



    # --- EPUB Processing Section ---
    with st.container():
        if epub_file:
            if st.button("Process EPUB File"):
                if process_epub_file(epub_file, top_n, base_directory, image_folder):
                    st.info("EPUB file processed.")

    # --- Character Summaries and Confirmation Section ---
    if st.session_state.get("top_n_summaries"):
        with st.container():
            display_character_summaries(top_n)
            confirm_character_summaries(base_directory)

    # --- Character Image Selection Section ---
    # Let the user pick which character image should appear first.
    if os.path.exists(image_folder):
        available_character_images = sorted([
            filename for filename in os.listdir(image_folder)
            if filename.lower().endswith('.png')
        ])
        if available_character_images:
            selected_character = st.selectbox(
                "Select the character image to appear first on the cover page",
                available_character_images
            )
            st.session_state["selected_character"] = selected_character
        else:
            st.warning("No character images found in the Character_Images folder.")


    # --- EPUB Reconstruction Section (only show after images are generated) ---
    if st.session_state.get("images_generated", False):
        with st.container():
            if st.button("Reconstruct EPUB with Generated Images"):
                reconstruct_epub()
 
        # Also show a separate tab/page for viewing all generated images.
        display_generated_images_page()

if __name__ == "__main__":
    main()
