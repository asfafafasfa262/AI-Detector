import joblib
import sys

# 載入模型 (注意：因為在根目錄執行，路徑要去掉 ../)
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
lr_model = joblib.load('models/logistic_regression_v1.pkl')

def predict(text):
    text_tfidf = tfidf.transform([text])
    prediction = lr_model.predict(text_tfidf)[0]
    prob = lr_model.predict_proba(text_tfidf)[0]
    label = "AI 生成" if prediction == 1 else "人類撰寫"
    return label, max(prob)

if __name__ == "__main__":
    # 讓你可以從終端機輸入文字測試
    input_text = input("請輸入一段文字進行偵測: ")
    res, score = predict(input_text)
    print(f"\n[偵測結果]: {res}")
    print(f"[信心程度]: {score:.4f}")
