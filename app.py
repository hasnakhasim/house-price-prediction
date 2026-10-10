
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="EstateWise | House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background: #FAF9F6;
    color: #24372F;
}

.block-container {
    max-width: 1200px;
    padding: 1rem 2.5rem 3rem;
}

/* Navigation */
.brand {
    color: #205747;
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
}

.nav-caption {
    color: #777E75;
    font-size: 12px;
    letter-spacing: 2px;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #163F33, #326B52);
    border-radius: 26px;
    padding: 45px 38px;
    color: white;
    margin: 25px 0 30px;
}

.hero-tag {
    display: inline-block;
    color: #D9E8D9;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    letter-spacing: 1px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 43px !important;
    line-height: 1.2 !important;
    margin: 20px 0 15px;
}

.hero p {
    color: #E0EAE1 !important;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
}

/* Section headings */
.section-title {
    color: #205747;
    font-size: 27px;
    font-weight: 800;
    margin: 20px 0 5px;
}

.section-subtitle {
    color: #7A8179;
    margin-bottom: 20px;
}

h2, h3 {
    color: #205747 !important;
}

p, label {
    color: #46574D;
}

/* Feature cards */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #EAE9E1;
    border-radius: 18px;
    padding: 24px 20px;
    min-height: 165px;
    box-shadow: 0 5px 20px rgba(35, 60, 40, 0.035);
}

.feature-icon {
    font-size: 29px;
    margin-bottom: 12px;
}

.feature-card h3 {
    color: #205747;
    font-size: 18px;
    margin-bottom: 8px;
}

.feature-card p {
    color: #777E75 !important;
    font-size: 13px;
    line-height: 1.6;
}

/* Input fields */
div[data-baseweb="input"] input {
    background: #FFFFFF !important;
    color: #24372F !important;
}

div[data-baseweb="select"] {
    background: #FFFFFF;
}

/* Golden-orange calculate button */
div.stButton > button {
    background: linear-gradient(90deg, #F59E0B, #EA580C);
    color: #FFFFFF !important;
    border: none;
    border-radius: 12px;
    min-height: 50px;
    font-weight: 700;
    font-size: 16px;
    width: 100%;
    box-shadow: 0 5px 14px rgba(234, 88, 12, 0.22);
    transition: all 0.25s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #D97706, #C2410C);
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(234, 88, 12, 0.30);
}

/* Property result card */
.result-panel {
    background: #E9F2E9;
    border: 1px solid #D2E3D2;
    border-left: 6px solid #477B61;
    border-radius: 18px;
    padding: 28px;
    margin-top: 22px;
}

.result-label {
    color: #55705D;
    font-size: 13px;
    letter-spacing: 1px;
}

.result-price {
    color: #205747;
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #E1E7DE;
}

hr {
    border-color: #E7E5DE;
}

/* Footer */
.footer {
    border-top: 1px solid #E7E5DE;
    padding-top: 22px;
    margin-top: 45px;
    color: #858B82;
    text-align: center;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("model/house_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Trained model not found. Run python train_model.py "
        "first to create model/house_model.pkl."
    )
    st.stop()


# ---------------- NAVIGATION ----------------

nav1, nav2 = st.columns([2, 1])

with nav1:
    st.markdown("""
    <div class="brand">⌂ EstateWise</div>
    <div class="nav-caption">SMART PROPERTY VALUATION</div>
    """, unsafe_allow_html=True)

with nav2:
    st.markdown("""
    <p style="text-align:right; padding-top:12px;">
        HOME &nbsp; · &nbsp; PRICE ESTIMATOR
    </p>
    """, unsafe_allow_html=True)


# ---------------- HERO HOMEPAGE ----------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">PROPERTY INTELLIGENCE</div>
    <h1>Every home has a value.<br>Discover yours.</h1>
    <p>
        Estimate a property's price using Machine Learning.
        Enter the basic property details and explore an
        estimated value in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- FEATURES ----------------

st.markdown("""
<div class="section-title">A smarter way to estimate</div>
<div class="section-subtitle">
    Simple property details. A machine learning estimate.
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📐</div>
        <h3>Property Size</h3>
        <p>Enter the built-up area to help estimate the
        property's potential value.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🏡</div>
        <h3>Home Details</h3>
        <p>Include the number of bedrooms, bathrooms
        and the property's age.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3>Price Estimate</h3>
        <p>Get a numerical price estimate from your
        trained regression model.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- PROPERTY ESTIMATOR ----------------

st.divider()

st.markdown("""
<div class="section-title">Estimate your property value</div>
<div class="section-subtitle">
    Enter your property information below.
</div>
""", unsafe_allow_html=True)

with st.container(border=True):

    st.subheader("🏠 Property Information")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Built-up area (sq. ft.)",
            min_value=100,
            max_value=20000,
            value=1200,
            step=100
        )

        bedrooms = st.selectbox(
            "Number of bedrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=2
        )

    with col2:
        bathrooms = st.selectbox(
            "Number of bathrooms",
            options=[1, 2, 3, 4, 5, 6],
            index=1
        )

        house_age = st.number_input(
            "Property age (years)",
            min_value=0,
            max_value=200,
            value=5,
            step=1
        )

    predict_clicked = st.button("Calculate Estimated Price")

    if predict_clicked:

        input_data = pd.DataFrame([{
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "house_age": house_age
        }])

        try:
            predicted_price = float(model.predict(input_data)[0])

            st.markdown(f"""
            <div class="result-panel">
                <div class="result-label">
                    ESTIMATED PROPERTY VALUE
                </div>
                <div class="result-price">
                    ₹{predicted_price:.2f} lakh
                </div>
                <p>
                    Estimate generated from the property
                    details you entered.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### Property Summary")

            r1, r2, r3, r4 = st.columns(4)

            r1.metric("Area", f"{area:,} sq. ft.")
            r2.metric("Bedrooms", bedrooms)
            r3.metric("Bathrooms", bathrooms)
            r4.metric("Property Age", f"{house_age} years")

            st.info(
                "This is an educational estimate based on your "
                "trained model. The sample dataset may not "
                "represent actual property market prices."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    EstateWise · House Price Prediction ·
    Built with Python, Streamlit and Machine Learning
</div>
""", unsafe_allow_html=True)
