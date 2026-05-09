import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

load_dotenv()
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Streamlit title ---
st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("📄 PDF İle Sohbet")
st.caption("PDF yükle, soru sor, RAG ile cevap al.")

# --- ChromaDB ---
class GeminiEmbedding(embedding_functions.EmbeddingFunction):
	def __call__(self, input):
		result = llm.models.embed_content(
			model="gemini-embedding-001",
			contents=input,
		)
		return [e.values for e in result.embeddings]

@st.cache_resource
def get_collection():
	chroma = chromadb.PersistentClient(path="./chroma_db")
	return chroma.get_or_create_collection(
		name="pdf_chunks",
		embedding_function=GeminiEmbedding(),
	)

collection = get_collection()

# --- PDF upload ---
uploaded = st.file_uploader("PDF dosyası seç", type="pdf")

if uploaded:
	if st.button("PDF'i işle"):
		with st.spinner("PDF okunuyor ve chunk'lanıyor..."): 
			reader = PdfReader(uploaded)

			splitter = RecursiveCharacterTextSplitter(
				chunk_size = 800,
				chunk_overlap = 150,
				separators = ["\n\n", "\n", ". ", " ", ""],
			)

			all_chunks = []
			all_metadata = []
			all_ids = []
			chunk_idx = 0

			for page_num, page in enumerate(reader.pages, start=1):
				page_text = page.extract_text()
				if not page_text.strip():
					continue
				page_chunks = splitter.split_text(page_text)
				for c in page_chunks:
					all_chunks.append(c)
					all_metadata.append({"page": page_num, "source": uploaded.name})
					all_ids.append(f"chunk_{chunk_idx}")
					chunk_idx += 1

			collection.add(
				documents=all_chunks,
				metadatas=all_metadata, 
				ids=all_ids,
			)
		st.success(f"{len(all_chunks)} chunk işlendi.")

# --- Soru sorma ---
question = st.text_input("Sorunu yaz:")

if question:
	with st.spinner("Cevap aranıyor..."):
		results = collection.query(query_texts=[question], n_results=8)
		context = "\n".join(f"- {d}" for d in results["documents"][0])

		prompt = f""" Aşağıdaki context'i kullanarak Türkçe cevap ver.
Eğer context'te yoksa "Belgede bilgi yok" de.                                                      
                                         
CONTEXT:
{context}
SORU: {question} 
"""

		response = llm.models.generate_content(
			model="gemini-2.5-flash",
			contents=prompt,
			config=types.GenerateContentConfig(
				temperature = 0.0,
				max_output_tokens = 400,
				thinking_config = types.ThinkingConfig(thinking_budget=0),
			),
		)
	st.markdown("### Cevap")
	st.write(response.text)

	with st.expander("Kullanılan context"):
		docs = results["documents"][0]
		metas = results["metadatas"][0]
		for i, (doc, meta) in enumerate(zip(docs, metas)):
			page = meta.get("page", "?")
			st.markdown(f"**[{i+1}] Sayfa {page}**")
			st.text(doc[:300] + "...")
