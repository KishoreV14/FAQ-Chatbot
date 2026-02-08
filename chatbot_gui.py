import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_reply():
    user_text = entry.get().lower()
    entry.delete(0, tk.END)

    if user_text == "":
        return

    chat_box.insert(tk.END, "You: " + user_text + "\n")

    user_vector = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()

    if similarity[0][best_match] > 0.2:
        chat_box.insert(tk.END, "Bot: " + answers[best_match] + "\n\n")
    else:
        chat_box.insert(tk.END, "Bot: Sorry, I don't understand that question.\n\n")

# GUI window
window = tk.Tk()
window.title("FAQ Chatbot")
window.geometry("500x400")

chat_box = tk.Text(window)
chat_box.pack(padx=10, pady=10)

# Initial greeting output
chat_box.insert(tk.END, "Bot: Hii 👋\n")
chat_box.insert(tk.END, "Bot: Hello! I am your FAQ assistant. Ask me anything.\n\n")

entry = tk.Entry(window, width=40)
entry.pack(pady=5)
send_button = tk.Button(window, text="Send", command=get_reply)
send_button.pack()

window.mainloop()