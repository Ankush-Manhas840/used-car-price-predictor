# Streamlit entry point for the live demo.
# The UI is static/index.html, embedded as a two-way component: the page sends the
# car's details to Python, Python predicts and sends the price back.
import streamlit as st
import streamlit.components.v1 as components
from carprice import HERE, options, predict, train

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
st.markdown("""<style>
  .block-container { padding-top: 2rem; max-width: 44rem; }
  #MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the model on first start (about 20 seconds)…")
def load_model():
    bundle = train()
    return bundle, options(bundle)


car_ui = components.declare_component("car_price_ui", path=str(HERE / "static"))

st.title("🚗 Used Car Price Predictor")
st.caption("Random forest trained on ~15,400 CarDekho listings · test R² 0.94 · typical error about ₹1 lakh · "
           "[How it was built](https://github.com/Ankush-Manhas840/used-car-price-predictor)")

bundle, opts = load_model()
request = car_ui(options=opts, result=st.session_state.get("result"), key="ui", default=None)

if request and request.get("id") != st.session_state.get("answered"):
    st.session_state.answered = request["id"]
    st.session_state.result = {"id": request["id"], "price": predict(bundle, request)}
    st.rerun()

st.caption("Not for luxury or exotic cars: brands with fewer than 10 listings were left out of training.")
