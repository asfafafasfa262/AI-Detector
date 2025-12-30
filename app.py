import streamlit as st
import joblib
import re
import torch
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 網頁基本配置 ---
st.set_page_config(page_title="AI 文本偵測終極專家系統", page_icon="🧠", layout="wide")

# --- 2. 載入所有模型資源 ---
@st.cache_resource
def load_all_assets():
    # 載入 TF-IDF 向量化器與機器學習模型
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    lr = joblib.load('models/logistic_regression_v1.pkl')
    rf = joblib.load('models/random_forest_v1.pkl')
    
    # 載入 BERT 深度學習模型 (請確保路徑與資料夾名稱正確)
    bert_path = "./models/bert_model" 
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    
    # 載入停用詞
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    return tfidf, lr, rf, bert_tokenizer, bert_model, stop_words

# --- 3. 工具函數 ---
def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def main():
    # --- 側邊欄：三路模型指標看板 ---
    with st.sidebar:
        st.header("📊 實驗室評測數據")
        st.caption("基於 380,000 筆資料測試結論")
        
        tab_lr, tab_rf, tab_bert = st.tabs(["LR", "RF", "BERT"])
        
        with tab_lr:
            st.subheader("Logistic Regression")
            st.metric("Accuracy", "98.0%")
            st.metric("Precision", "98.1%")
            st.metric("Recall", "97.9%")
            st.metric("F1-score", "0.980")
            
        with tab_rf:
            st.subheader("Random Forest")
            st.metric("Accuracy", "96.5%") 
            st.metric("Precision", "96.8%")
            st.metric("Recall", "96.2%")
            st.metric("F1-score", "0.965")
            
        with tab_bert:
            st.subheader("🚀 DistilBERT")
            st.metric("Accuracy", "98.92%", delta="最高準確")
            st.metric("Precision", "97.67%")
            st.metric("Recall", "99.60%", delta="最強偵測力")
            st.metric("F1-score", "0.986")

        st.divider()
        st.info("""
        **🔍 偵測原理對比：**
        - **LR/RF**: 基於單詞頻率與統計特徵。
        - **BERT**: 基於語意上下文與注意力機制。
        """)

    # --- 主畫面介面 ---
    st.title("🛡️ AI vs. 人類文本偵測系統 (多模型整合版)")
    st.markdown("本系統整合了傳統統計學習與最新深度學習技術，提供全方位的文本來源分析。")
    
    user_input = st.text_area("請輸入欲偵測之英文文本 (建議 50 字以上):", height=200)
    THRESHOLD = 0.7

    try:
        tfidf, lr, rf, b_tokenizer, b_model, stop_words = load_all_assets()

        if st.button("🚀 執行三模型深度診斷"):
            if user_input.strip():
                # --- A. 傳統模型運算 ---
                cleaned = clean_text(user_input, stop_words)
                vec = tfidf.transform([cleaned])
                lr_prob = lr.predict_proba(vec)[0][1]
                rf_prob = rf.predict_proba(vec)[0][1]

                # --- B. BERT 模型運算 ---
                with st.spinner('BERT 正在進行深層語意理解...'):
                    inputs = b_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = b_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        bert_prob = probs[0][1].item()

                # --- C. 結果顯示 ---
                st.divider()
                st.subheader("🎯 各模型判定結果")
                cols = st.columns(3)
                
                def display_box(title, prob, col):
                    with col:
                        st.write(f"**{title}**")
                        if prob >= THRESHOLD:
                            st.error("🤖 AI 生成")
                        elif prob <= (1 - THRESHOLD):
                            st.success("✍️ 人類撰寫")
                        else:
                            st.warning("❓ 判定模糊")
                        st.write(f"AI 可能性: {prob*100:.1f}%")

                display_box("Logistic Regression", lr_prob, cols[0])
                display_box("Random Forest", rf_prob, cols[1])
                display_box("DistilBERT (深度學習)", bert_prob, cols[2])

                # --- D. 綜合評估結論 ---
                st.divider()
                ai_votes = sum([1 for p in [lr_prob, rf_prob, bert_prob] if p >= THRESHOLD])
                human_votes = sum([1 for p in [lr_prob, rf_prob, bert_prob] if p <= (1 - THRESHOLD)])
                
                st.subheader("📝 綜合診斷報告")
                if ai_votes >= 2:
                    st.error(f"🚩 偵測警告：經多數模型共識（{ai_votes}/3），此文本極大機率為 AI 生成內容。")
                elif human_votes >= 2:
                    st.success(f"✅ 偵測結果：經多數模型共識（{human_votes}/3），此文本展現高度人類撰寫特徵。")
                else:
                    st.warning("⚠️ 系統警示：模型判定結果存在分歧，建議參考各模型信心度並結合人工審核。")

            else:
                st.warning("⚠️ 請輸入內容後再點擊按鈕。")

    except Exception as e:
        st.error(f"系統載入錯誤，請確認模型檔案路徑。錯誤訊息: {e}")

if __name__ == "__main__":
    main()
