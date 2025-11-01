import streamlit as st
import torch
import json
from PIL import Image
from model import FlowerClassifier
from torchvision import transforms
from pathlib import Path

# setting the page title
st.set_page_config(page_title="flower classifier", page_icon="🌸", layout="centered")

# loading json file which contains list of flower classes
current_dir = Path(__file__).parent
json_path = current_dir / "cat_to_name.json"

with open(json_path,'r') as f:
    cat_to_name = json.load(f)

st.title("flower species classifier")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ about")
    st.subheader("🏗️ model architexture")
    st.write("ResNet18 - transfer learning; pre trained on ImageNet")
    st.write("fined tuned on final layer only - 102 flower species")
    
    st.markdown("---")

    st.subheader("📊 performance")
    st.write("test set : 81 % accuracy")
    st.write("real world : variable performance 🥲")

    st.markdown("---")

    st.subheader("⚠️ known limitations")
    st.write("test set accuracy ≠ real world accuracy")

# using this wrapper; so that the function runs once and stays in the cache
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowerClassifier(num_classes=102)
    # loading the best model
    current_dir = Path(__file__).parent
    model_path = current_dir / "models" / "best_model.pth"

    checkpoint = torch.load(model_path, map_location=device) # loads gpu-trained model on cpu
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device

# applying the transforms to the uploaded image
preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.CenterCrop((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # ImageNet normalization

model, device = load_model()

uploaded_file = st.file_uploader("choose an image : ", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:    
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="uploaded image", use_container_width=True)

    with col2:
        if st.button("predict") or st.session_state.get("predict_clicked", False):
            with st.spinner("analysing flower ..."):
                tensor_img = preprocess(img).unsqueeze(0)
                tensor_img = tensor_img.to(device)
                output = model(tensor_img)

                # same as torch.max() -> dimension 1
                probs = torch.softmax(output, dim=1) # shape -> (batch size, classes)
                # gets the top 'k' highest values from dimension 1- > classes
                top5_probs, top5_indices = probs.topk(5, dim=1)

                top5_probs = top5_probs[0].tolist()
                top5_indices = top5_indices[0].tolist()

                st.subheader("top 5 predictions : ")
                for i in range(5):
                    idx = top5_indices[i]
                    prob = top5_probs[i]
                    flower_name = cat_to_name[str(idx)]

                    st.write(f"**{i+1}. {flower_name}** - {prob*100:.2f}%")
                    st.progress(prob)