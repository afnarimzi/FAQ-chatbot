from flask import Flask,request,jsonify,render_template
from gensim.models import Word2Vec
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt_tab')

app=Flask(__name__)
nlp=spacy.load("en_core_web_sm")
faq_data = {
    "What are your working hours?": "Our shop is open from 9 AM to 9 PM every day.",
    "When does the shop open?": "The shop opens at 9 AM.",
    "When does the shop close?": "The shop closes at 9 PM.",
    "Are you open on weekends?": "Yes, we are open every day, including weekends.",
    "Are you open on holidays?": "Yes, we are open on most holidays",
    "Where is the shop location?": "We are located at 123 Fashion Street, New York.",
    "Do you offer home delivery?": "Yes, we provide home delivery within the city.",
    "What type of clothes do you sell?": "We sell men's, women's, and kids' fashion wear.",
    "How can I contact customer support?": "You can contact us at timelesstrends@example.com or call 123-456-7890.",
    "Is there a return policy?": "Yes, you can return items within 30 days with a receipt.",
    "Do you have offers?": "Yes, we offer seasonal offers.",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and Apple Pay.",
    "Do you offer international shipping?": "Yes, we ship worldwide with standard shipping rates.",
}
 
faq_questions=[nltk.word_tokenize(q.lower()) for q in faq_data.keys()]
word2vec_model=Word2Vec(faq_questions,vector_size=50,window=2,min_count=1,workers=4)

def get_sentence_vector(sentence):
    words=nltk.word_tokenize(sentence.lower())
    vectors=[word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(vectors,axis=0) if vectors else np.zeros(50)

def find_best_match(user_question):
    user_vector=get_sentence_vector(user_question)
    faq_vectors=[get_sentence_vector(q) for q in faq_data.keys()]
    similarities=cosine_similarity([user_vector],faq_vectors)[0]
    best_match_index=np.argmax(similarities)
    best_question=list(faq_data.keys())[best_match_index]
    return faq_data[best_question]

# Flask route for frontend
@app.route("/")
def home():
    return render_template("chatbot.html")

# API endpoint for chatbot response
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")  # Get user message from JSON
    response = find_best_match(user_input)  # Find best match
    return jsonify({"response": response})  # Return chatbot response

if __name__ == "__main__":
    app.run(debug=True)