import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import os
import base64

import googlemaps
import folium
from streamlit_folium import st_folium

# --- NEW: Import the custom geolocation component ---
from geolocation_component import get_geolocation # Assuming this file is in the same directory

# --- API Key Loading ---
# Remove the os.getenv("GOOGLE_PLACES_API_KEY") line
# GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") # REMOVE THIS LINE

# Directly try to load from Streamlit secrets
GOOGLE_PLACES_API_KEY = None
try:
    # Attempt to load from Streamlit secrets
    if "google" in st.secrets and "places_api_key" in st.secrets["google"]:
        GOOGLE_PLACES_API_KEY = st.secrets["google"]["places_api_key"]
except Exception as e:
    st.error(f"Error loading API key from Streamlit secrets: {e}")
    pass # Keep GOOGLE_PLACES_API_KEY as None if there's an error

gmaps = None # Initialize gmaps to None
if not GOOGLE_PLACES_API_KEY:
    st.error("🚨 Google Places API Key not found. Hospital/Doctor finder and related features will not work. 🚨")
    st.info("For local development, ensure you have a '.streamlit/secrets.toml' file in your project root with the correct key structure.")
    st.info("For Streamlit Cloud deployment, configure your '.streamlit/secrets.toml' file.")
else:
    gmaps = googlemaps.Client(key=GOOGLE_PLACES_API_KEY)


# --- Caching Helpers ---
@st.cache_resource
def load_model():
    """Loads the TensorFlow Keras model, cached for efficiency."""
    try:
        model_path = "Trained_Model.keras"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}. Please ensure 'Trained_Model.keras' is in the correct directory.")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Could not load the model. Error: {e}")
        return None

@st.cache_data
def get_base64_image(path): # Renamed for clarity, was get_base64
    """Encodes a file to base64 for embedding in CSS."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: File not found at {path}. Please ensure image files are in the correct directory.")
        return ""

@st.cache_data
def preprocess_image_for_model(image_path): # Renamed for clarity, was preprocess_image
    """Loads and preprocesses an image for model prediction."""
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    return preprocess_input(np.expand_dims(x, 0))

def model_prediction(test_image_path):
    model = load_model()
    if model is None:
        return -1

    try:
        with st.spinner("Analyzing image... Please wait."):
            preds = model.predict(preprocess_image_for_model(test_image_path))[0]
            names = ['CNV','DME','DRUSEN','NORMAL']
            st.session_state.confidence_scores = {names[i]: float(preds[i]) for i in range(4)}
            return np.argmax(preds)
    except Exception as e:
        st.error(f"Error during image preprocessing or prediction: {e}")
        return -1

@st.cache_data(show_spinner="Getting coordinates...")
def get_coordinates(address):
    """Converts an address to latitude and longitude using Geocoding API."""
    if gmaps is None:
        return None, None
    try:
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            lat = geocode_result[0]['geometry']['location']['lat']
            lon = geocode_result[0]['geometry']['location']['lng']
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.error(f"Error getting coordinates for '{address}': {e}")
        return None, None

@st.cache_data(show_spinner="Searching for places...")
def search_hospitals_doctors(query, lat, lon, radius_meters=50000):
    """Searches for places using Google Places API (Text Search for flexibility)."""
    if gmaps is None:
        return []
    try:
        # Include fields that give us more data, but be mindful of costs
        # 'opening_hours' is often in 'place_details', not directly in 'places' (text search)
        # We can make a separate call for details if needed, but for now, rely on basic info.
        places_result = gmaps.places(query=query, location=(lat, lon), radius=radius_meters)
        return places_result.get('results', [])
    except Exception as e:
        st.error(f"Error searching for places: {e}")
        return []

@st.cache_data(show_spinner="Calculating travel times...")
def get_distance_matrix(origins, destinations, mode="driving"):
    """Calculates distances and travel times using Distance Matrix API."""
    if gmaps is None:
        return None
    try:
        # Origins can be a list of addresses or lat/lon pairs
        # Destinations can be a list of addresses or lat/lon pairs
        result = gmaps.distance_matrix(origins, destinations, mode=mode)
        return result
    except Exception as e:
        st.error(f"Error getting distance matrix: {e}")
        return None


# --- Disease Descriptions ---
cnv = """
### Choroidal Neovascularization (CNV)
CNV is a severe condition where new, abnormal blood vessels grow beneath the retina. These vessels are fragile and can leak fluid or blood, leading to significant vision loss. It's often associated with wet Age-related Macular Degeneration (AMD).
**Symptoms:** Blurred vision, distorted vision (straight lines appear wavy), a blind spot in central vision.
**Importance of Early Detection:** Early diagnosis is crucial for effective treatment. Treatments often involve anti-VEGF injections, which can help stop the growth of new vessels and reduce leakage, preserving vision.
"""
dme = """
### Diabetic Macular Edema (DME)
DME is a complication of diabetes, characterized by swelling in the macula (the central part of the retina responsible for sharp, detailed vision) due to leaky blood vessels. It's a common cause of vision loss in people with diabetes.
**Symptoms:** Blurred vision, difficulty reading, colors appearing faded.
**Importance of Early Detection:** Good blood sugar control is vital for preventing and managing DME. Treatments include anti-VEGF injections, laser therapy, and steroid implants, which aim to reduce swelling and improve vision. Regular eye exams are essential for diabetics.
"""
drusen = """
### Drusen (Early Age-related Macular Degeneration - AMD)
Drusen are yellow deposits of fatty proteins that accumulate under the retina. Small, scattered drusen are common with age and typically don't cause vision problems. However, larger or numerous drusen can be an early sign of Age-related Macular Degeneration (AMD), a leading cause of vision loss.
**Symptoms:** Often asymptomatic in early stages. May experience slight blurring or distortion of vision as it progresses.
**Importance of Early Detection:** While early AMD may not require immediate treatment, monitoring is important. Lifestyle changes (e.g., healthy diet, not smoking) and AREDS supplements (for specific types of AMD) can help slow progression. Regular dilated eye exams are recommended.
"""
normal = """
### Normal Retina
A normal retina indicates no signs of the specific eye diseases (CNV, DME, Drusen) detectable by this model. This is a healthy finding, suggesting good retinal health based on the provided OCT image.
**Recommendation:** Continue with routine eye examinations as recommended by your ophthalmologist to monitor overall eye health.
"""

# --- Background Setup ---
def set_background(path):
    """Sets the background image and applies global CSS for the Streamlit app."""
    b64 = get_base64_image(path)
    if b64:
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{b64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            /* Reduce default padding/margin for main content and sidebar */
            .st-emotion-cache-1pxazr7, .st-emotion-cache-z5fcl4, .st-emotion-cache-1jmve30, .st-emotion-cache-13ln4gm {{
                padding-top: 1rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 1rem;
                margin-top: 0rem;
            }}
            [data-testid="stHeader"] {{
                background-color: rgba(255, 255, 255, 0.7);
                box-shadow: 0 2px 4px rgba(0,0,0,.1);
            }}
            [data-testid="stSidebar"] {{
                background-color: rgba(255, 255, 255, 0.8);
                box-shadow: 2px 0 4px rgba(0,0,0,.1);
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 16px;
                transition: background-color 0.3s ease, box-shadow 0.3s ease;
                border: none;
            }}
            .stButton>button:hover {{
                background-color: #45a049;
                box-shadow: 0 4px 8px rgba(0,0,0,.2);
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2E8B57;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .main .block-container {{
                color: #333333;
                line-height: 1.6;
            }}
            .stAlert {{
                border-radius: 8px;
            }}
            .streamlit-expanderHeader {{
                background-color: #e0ffe0;
                border-left: 5px solid #4CAF50;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }}
            .streamlit-expanderContent {{
                background-color: #f0fff0;
                padding: 15px;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }}
            .find-hospitals-section {{
                background-color: rgba(255, 255, 255, 0.9);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .stRadio > div {{ margin-bottom: 0.5rem; }}
            .stTextInput > div {{ margin-bottom: 0.5rem; }}
            .st-emotion-cache-pe2l5p {{
                border-radius: 10px;
                overflow: hidden;
                margin-top: 15px;
                margin-bottom: 15px;
            }}
            </style>
        """, unsafe_allow_html=True)


# --- Footer ---
def render_footer():
    """Renders a fixed footer at the bottom of the app."""
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: rgba(240, 242, 246, 0.9);
            color: #555;
            text-align: center;
            padding: 10px 0;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
            z-index: 999;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
            /* Add this line: */
            flex-shrink: 0; /* Prevents the footer from shrinking in a flex container */
        }
        .main {
            padding-bottom: 70px; /* Adjust this value if your footer's height changes */
        }
        </style>
        <div class="footer">
            © 2025 Medicare – Eye Disease Detector. For educational use only. Developed by <a href="https://github.com/MTSAHU" target="_blank">Amit</a>. 
        </div>
    """, unsafe_allow_html=True)

# --- "Back to Top" button utility ---
def add_back_to_top_button():
    st.markdown("""
        <style>
        .back-to-top {
            position: fixed;
            bottom: 80px;
            right: 20px;
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 50%;
            text-align: center;
            cursor: pointer;
            font-size: 1.5em;
            line-height: 1;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }
        .back-to-top:hover {
            background-color: #45a049;
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        </style>
        <div class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">⬆️</div>
    """, unsafe_allow_html=True)


# --- Page Render Functions ---
def render_home():
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1.  **Choroidal Neovascularization (CNV)**
    - Neovascular membrane with subretinal fluid

2.  **Diabetic Macular Edema (DME)**
    - Retinal thickening with intraretinal fluid

3.  **Drusen (Early AMD)**
    - Presence of multiple drusen deposits

4.  **Normal Retina**
    - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

---

#### **Contact Us**

Have questions or need assistance? [Contact our support team](#) for more information on how to use the platform or integrate it into your clinical practice.

    """)

def render_about():
    st.header("About This Project")
    st.markdown("""
        This project aims to provide an accessible tool for preliminary analysis of Optical Coherence Tomography (OCT) retinal images.
        Leveraging a deep learning model trained on a large dataset of diverse OCT scans,
        we offer an automated way to classify images into categories of healthy retinas or those showing signs of CNV, DME, or Drusen.
        """)

    st.subheader("Our Mission")
    st.markdown("""
        To assist in early detection and raise awareness about critical eye conditions, making advanced diagnostic tools more approachable for educational purposes.
        """)

    st.subheader("Technology Stack")
    st.markdown("""
        - **Frontend:** Streamlit for interactive web application development.
        - **Backend:** TensorFlow/Keras for building and deploying the deep learning model.
        - **APIs:** Google Maps Platform APIs (Geocoding, Places, Distance Matrix) for location-based services.
        - **Data:** A comprehensive dataset of OCT images curated and verified by medical professionals.
        """)

    st.subheader("The Team")
    st.info("Developed by AMIT for the **ML for Eye Disease Identification** capstone project as an intern at **NIT Rourkela**.") # Placeholder

    st.markdown("---")
    st.subheader("Disclaimer")
    st.warning("""
        This application is developed for educational and demonstration purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)

def render_disease_identification():
    st.header("Upload OCT Image for Diagnosis")

    # File uploader
    test_image_upload = st.file_uploader("Upload your Image:", type=["jpg", "png", "jpeg"], key="oct_image_uploader")

    # --- Image Upload and Prediction Logic ---
    if test_image_upload is not None:
        # Check if a new image is uploaded or if the existing one is different
        if st.session_state.get('uploaded_image_bytes') is None or \
           (test_image_upload.getvalue() != st.session_state.uploaded_image_bytes and test_image_upload.name != st.session_state.uploaded_image_name):
            
            st.session_state.uploaded_image_bytes = test_image_upload.getvalue()
            st.session_state.uploaded_image_name = test_image_upload.name
            st.session_state.prediction_made = False
            st.session_state.confidence_scores = {}
            st.session_state.prediction_result_index = -1
            st.session_state.search_results = None # Clear previous search results

        st.image(st.session_state.uploaded_image_bytes, caption=f"Uploaded OCT Image: {st.session_state.uploaded_image_name}", use_container_width=True)
        
        if not st.session_state.prediction_made:
            if st.button("🔍 **Analyze Image**", help="Click to get a diagnosis for the uploaded image."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.uploaded_image_name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(st.session_state.uploaded_image_bytes)
                    temp_file_path = tmp_file.name
                
                st.session_state.prediction_result_index = model_prediction(temp_file_path)
                st.session_state.prediction_made = True

                os.unlink(temp_file_path) # Clean up temp file
                st.rerun() # Rerun to show prediction results
        
    elif st.session_state.get('uploaded_image_bytes') is None:
        st.info("Upload an OCT image (JPG, PNG, JPEG) to get a diagnosis and find nearby facilities.")


    # --- Display Prediction Results and Hospital Finder ---
    if st.session_state.prediction_made and st.session_state.prediction_result_index != -1:
        result_index = st.session_state.prediction_result_index

        class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        st.success(f"Model predicts: **{class_name[result_index]}**")

        st.subheader("Confidence Scores:")
        if 'confidence_scores' in st.session_state and st.session_state.confidence_scores:
            cols = st.columns(len(st.session_state.confidence_scores))
            sorted_scores = sorted(st.session_state.confidence_scores.items(), key=lambda item: item[1], reverse=True)
            for i, (cls, conf) in enumerate(sorted_scores):
                with cols[i]:
                    st.metric(label=cls, value=f"{conf*100:.2f}%")
        else:
            st.info("Confidence scores not available.")

        st.markdown("---")
        with st.expander("Understanding the Diagnosis", expanded=True):
            if result_index == 0: st.markdown(cnv)
            elif result_index == 1: st.markdown(dme)
            elif result_index == 2: st.markdown(drusen)
            elif result_index == 3: st.markdown(normal)

        st.markdown("---")

        # --- Hospital and Doctor Finder Section ---
        st.markdown('<div class="find-hospitals-section">', unsafe_allow_html=True)
        st.subheader("Find Nearby Hospitals and Eye Doctors")
        
        st.info("This feature helps you locate medical facilities based on your diagnosis. Always consult a healthcare professional.")

        if gmaps is None:
            st.error("Hospital/Doctor finder is unavailable due to missing Google API Key configuration.")
        else:
            # Geolocation component call - wrap in a container
            geolocation_placeholder = st.empty()
            with geolocation_placeholder:
                # This will run the JS to get location and update session state variables
                get_geolocation()
            
            # Display status of live location
            if st.session_state.user_latitude is not None and st.session_state.user_longitude is not None:
                geolocation_placeholder.success(f"Live location acquired: Lat {st.session_state.user_latitude:.4f}, Lon {st.session_state.user_longitude:.4f}")
            else:
                geolocation_placeholder.info("Waiting for live location permission. If permission is denied or times out, you can enter a location manually.")
            
            search_option_labels = ["By entering a specific location"]
            if st.session_state.user_latitude is not None and st.session_state.user_longitude is not None:
                search_option_labels.insert(0, "Near my current (live) location")
                
            search_option = st.radio(
                "How would you like to search for facilities?",
                search_option_labels,
                key="search_location_option"
            )

            search_lat, search_lon = None, None
            search_location_display = ""

            if search_option == "Near my current (live) location":
                search_lat = st.session_state.user_latitude
                search_lon = st.session_state.user_longitude
                search_location_display = "Your Current (Live) Location"
            else: # "By entering a specific location"
                user_location_text = st.text_input("Enter Country, City, Address, or Landmark:", st.session_state.get("last_manual_location", "Bengaluru, India"), key="user_location_input")
                if user_location_text:
                    st.session_state.last_manual_location = user_location_text
                    search_lat, search_lon = get_coordinates(user_location_text)
                    if search_lat and search_lon:
                        search_location_display = user_location_text
                        st.success(f"Searching near: {search_location_display}")
                    else:
                        st.warning("Could not find coordinates for the entered location. Please refine your input.")

            if search_lat and search_lon:
                st.markdown("---")
                search_query = st.text_input("What are you looking for? (e.g., 'eye doctor', 'ophthalmologist', 'hospital')", st.session_state.get("last_search_query", "eye doctor"), key="search_query_input")
                st.session_state.last_search_query = search_query

                search_radius_km = st.slider("Search Radius (km)", 1, 100, st.session_state.get("last_search_radius", 10), key="search_radius_slider")
                st.session_state.last_search_radius = search_radius_km
                search_radius_meters = search_radius_km * 1000

                if st.button("Search Hospitals/Doctors", key="search_facilities_button"):
                    places = search_hospitals_doctors(search_query, search_lat, search_lon, search_radius_meters)
                    
                    # Store data needed to re-render consistent map/list
                    st.session_state.search_results = places
                    st.session_state.search_location_for_map = (search_lat, search_lon)
                    st.session_state.search_location_display_text = search_location_display
                    st.session_state.search_query_text = search_query
                    st.session_state.search_radius_km_for_display = search_radius_km # For clarity in display

                    # Calculate distances/durations only if places are found
                    if places:
                        origin_coords = (search_lat, search_lon)
                        destination_coords = [(p['geometry']['location']['lat'], p['geometry']['location']['lng']) for p in places]
                        
                        distance_matrix_result = get_distance_matrix([origin_coords], destination_coords)
                        
                        st.session_state.distance_matrix_info = {}
                        if distance_matrix_result and distance_matrix_result['rows']:
                            for i, element in enumerate(distance_matrix_result['rows'][0]['elements']):
                                if element['status'] == 'OK':
                                    st.session_state.distance_matrix_info[places[i]['place_id']] = {
                                        'distance': element['distance']['text'],
                                        'duration': element['duration']['text']
                                    }
                                else:
                                    st.session_state.distance_matrix_info[places[i]['place_id']] = {
                                        'distance': 'N/A',
                                        'duration': 'N/A'
                                    }
                        else:
                            st.info("Could not calculate travel times/distances.")
                    else:
                        st.session_state.distance_matrix_info = {} # Clear if no places found
                    st.rerun() # Rerun to display results


                # Display search results if available
                if st.session_state.get('search_results') is not None and \
                   st.session_state.get("search_location_display_text") == search_location_display and \
                   st.session_state.get("search_query_text") == search_query and \
                   st.session_state.get("search_radius_km_for_display") == search_radius_km: # Check all params
                    
                    places = st.session_state.search_results
                    
                    if places:
                        st.subheader(f"Found {len(places)} results for '{search_query}' near {search_location_display} (within {search_radius_km} km):")

                        m = folium.Map(location=st.session_state.search_location_for_map, zoom_start=12)
                        
                        # Add marker for the searched origin
                        folium.Marker(
                            location=st.session_state.search_location_for_map,
                            popup=f"**Searched Location:**<br>{search_location_display}",
                            icon=folium.Icon(color="red", icon="location-dot", prefix="fa") # Use a location icon
                        ).add_to(m)

                        for place in places:
                            name = place.get("name", "N/A")
                            address = place.get("formatted_address", "N/A")
                            place_lat = place["geometry"]["location"]["lat"]
                            place_lon = place["geometry"]["location"]["lng"]

                            # Get travel info if available
                            travel_info = st.session_state.distance_matrix_info.get(place.get("place_id"), {})
                            distance = travel_info.get('distance', 'N/A')
                            duration = travel_info.get('duration', 'N/A')

                            popup_html = f"<b>{name}</b><br>{address}"
                            if distance != 'N/A' and duration != 'N/A':
                                popup_html += f"<br>Distance: {distance}<br>Duration: {duration}"

                            folium.Marker(
                                location=[place_lat, place_lon],
                                popup=folium.Popup(popup_html, max_width=300), # Use Popup for more control
                                icon=folium.Icon(color="blue", icon="hospital", prefix="fa")
                            ).add_to(m)

                        # Display the map using streamlit_folium
                        st_folium(m, width="100%", height=500, key="folium_map")

                        st.subheader("List of Facilities:")
                        for i, place in enumerate(places):
                            name = place.get("name", "N/A")
                            address = place.get("formatted_address", "N/A")
                            rating = place.get("rating", "N/A")
                            user_ratings_total = place.get("user_ratings_total", "N/A")
                            business_status = place.get("business_status", "UNKNOWN").replace('_', ' ').title()

                            travel_info = st.session_state.distance_matrix_info.get(place.get("place_id"), {})
                            distance = travel_info.get('distance', 'N/A')
                            duration = travel_info.get('duration', 'N/A')

                            st.markdown(f"**{i+1}. {name}**")
                            st.write(f"Address: {address}")
                            st.write(f"Status: {business_status}")
                            if rating != "N/A":
                                st.write(f"Rating: {rating} ({user_ratings_total} reviews)")
                            if distance != 'N/A':
                                st.write(f"Approx. Distance: {distance}")
                            if duration != 'N/A':
                                st.write(f"Approx. Travel Time (Driving): {duration}")

                            place_id = place.get("place_id")
                            if place_id:
                                st.markdown(f"[Get Directions on Google Maps](https://www.google.com/maps/dir/?api=1&destination=place_id:{place_id}&origin={search_lat},{search_lon})", unsafe_allow_html=True)
                            st.markdown("---")

                    else:
                        st.warning(f"No results found for '{search_query}' near {search_location_display}. Try increasing the search radius or refining your query.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_contact():
    st.header("✉️ Contact Us")
    st.markdown("""
        Have questions, feedback, or suggestions? We'd love to hear from you!
        """)

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.text_input("Subject")
        message = st.text_area("Your Message", height=150)

        submitted = st.form_submit_button("Send Message")

        if submitted:
            if not name or not email or not subject or not message:
                st.error("Please fill in all fields.")
            elif "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                # In a real application, you would send this data to a backend service (e.g., email service, database)
                st.success("Thank you for your message! We will get back to you shortly.")
                st.write(f"**Name:** {name}")
                st.write(f"**Email:** {email}")
                st.write(f"**Subject:** {subject}")
                st.write(f"**Message:** {message}")
                # You might want to clear the form fields after submission, but Streamlit doesn't natively
                # support clearing text_input/text_area easily without a full rerun or session state manipulation.


# --- Main App Logic ---
def main():
    st.set_page_config(
        page_title="Medicare Eye Detector",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state variables
    if 'confidence_scores' not in st.session_state: st.session_state.confidence_scores = {}
    if 'uploaded_image_bytes' not in st.session_state: st.session_state.uploaded_image_bytes = None
    if 'uploaded_image_name' not in st.session_state: st.session_state.uploaded_image_name = None
    if 'prediction_result_index' not in st.session_state: st.session_state.prediction_result_index = -1
    if 'prediction_made' not in st.session_state: st.session_state.prediction_made = False
    if 'search_results' not in st.session_state: st.session_state.search_results = None
    if 'search_location_for_map' not in st.session_state: st.session_state.search_location_for_map = None
    if 'search_location_display_text' not in st.session_state: st.session_state.search_location_display_text = ""
    if 'search_query_text' not in st.session_state: st.session_state.search_query_text = ""
    if 'search_radius_km_for_display' not in st.session_state: st.session_state.search_radius_km_for_display = 10
    if 'user_latitude' not in st.session_state: st.session_state.user_latitude = None
    if 'user_longitude' not in st.session_state: st.session_state.user_longitude = None
    if 'distance_matrix_info' not in st.session_state: st.session_state.distance_matrix_info = {}

    # Set background
    background_image_path = "copy-space-medical-workspace.jpg" # Ensure this image exists
    if os.path.exists(background_image_path):
        set_background(background_image_path)
    else:
        st.warning(f"Background image not found at: {background_image_path}. Please check the path. Using a default background color.")
        st.markdown("<style>.stApp { background-color: #f0f2f6; }</style>", unsafe_allow_html=True)

    # Manual top bar with logo and title
    logo_path = "copy-space-medical-workspace.jpg" # Ensure this image exists
    logo_html = ""
    if os.path.exists(logo_path):
        logo_base64_data = get_base64_image(logo_path)
        if logo_base64_data:
            logo_html = f'<img src="data:image/jpg;base64,{logo_base64_data}" width="50" style="margin-right:15px; border-radius: 5px;"/>'
    else:
        st.warning(f"Logo image not found at: {logo_path}. Please ensure it's in the correct directory.")

    st.markdown(f"""
        <div style="display:flex; align-items:center; padding:0.5rem; background-color: rgba(255, 255, 255, 0.7); border-radius: 10px; margin-bottom: 20px;">
            {logo_html}
            <h1 style="margin:0; color: #2E8B57;">Medicare Eye Disease Detector</h1>
        </div>
    """, unsafe_allow_html=True)

    # --- Native multipage setup using st.navigation ---
    pages = [
        st.Page(render_home, title="Home", icon="🏠"),
        st.Page(render_about, title="About Project", icon="ℹ️"), # New 'About Project' page
        st.Page(render_disease_identification, title="Disease Identification", icon="🔬"),
        st.Page(render_contact, title="Contact Us", icon="✉️") # New 'Contact Us' page
    ]

    pg = st.navigation(pages)
    pg.run()
    
    # The footer and back-to-top button should always be rendered
    render_footer()
    add_back_to_top_button()


if __name__ == "__main__":
    main()