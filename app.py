import os
import time
import datetime as dt
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types, errors

load_dotenv()
llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Streamlit title ---
st.set_page_config(page_title="PDF Chat", page_icon="📄")
st.title("📄 PDF İle Sohbet")
st.caption("PDF yükle, soru sor, RAG ile cevap al.")

# Demo limitleri — free tier kotasını korumak için
MAX_QUERIES_PER_SESSION = 5    # her tarayıcı oturumu için
MAX_DAILY_QUERIES = 50         # tüm kullanıcılar için günlük toplam

if "query_count" not in st.session_state:
	st.session_state.query_count = 0


@st.cache_resource
def get_daily_counter():
	# Tüm session'lar arası paylaşılan singleton
	return {"date": dt.date.today(), "count": 0}


def check_daily_limit():
	counter = get_daily_counter()
	today = dt.date.today()
	if counter["date"] != today:
		counter["date"] = today
		counter["count"] = 0
	return counter

# --- ChromaDB ---
class GeminiEmbedding(embedding_functions.EmbeddingFunction):
	def __call__(self, input):
		# Gemini batch limit 100, biz 50'lik batch'lerle güvenli gidiyoruz
		# 429 rate limit gelirse 30 sn bekleyip tekrar deniyoruz
		embeddings = []
		batch_size = 50
		for i in range(0, len(input), batch_size):
			batch = input[i:i + batch_size]
			for attempt in range(3):
				try:
					result = llm.models.embed_content(
						model="gemini-embedding-001",
						contents=batch,
					)
					embeddings.extend([e.values for e in result.embeddings])
					break
				except errors.ClientError as e:
					if e.code == 429 and attempt < 2:
						st.toast(f"⏳ Yoğunluk var, 30 saniye bekleniyor... (deneme {attempt + 2}/3)", icon="⏳")
						time.sleep(30)
						continue
					raise
		return embeddings

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
st.caption("📌 Önerilen: 5-15 sayfa, max 2MB · Büyük PDF'ler işlem süresini uzatır")

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

if uploaded:
	if uploaded.size > MAX_FILE_SIZE:
		st.error(f"⚠️ Dosya çok büyük ({uploaded.size / 1024 / 1024:.1f}MB). Lütfen 2MB altı bir PDF deneyin.")
		st.stop()

	if st.button("PDF'i işle"):
		with st.spinner("PDF okunuyor ve chunk'lanıyor..."):
			# Önce eski PDF'in chunk'larını temizle
			existing = collection.get()["ids"]
			if existing:
				collection.delete(ids=existing)

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
st.caption(f"💬 Bu oturumda kalan soru hakkı: {MAX_QUERIES_PER_SESSION - st.session_state.query_count}/{MAX_QUERIES_PER_SESSION}")


@st.cache_data(ttl=3600)
def ask_llm(question: str, context: str) -> str:
	# Aynı soru-context tekrar gelirse cache'den dön, yeni API çağrısı yapma
	prompt = f"""Aşağıdaki context'i kullanarak Türkçe cevap ver.
Eğer context'te yoksa "Belgede bilgi yok" de.

CONTEXT:
{context}

SORU: {question}
"""
	response = llm.models.generate_content(
		model="gemini-2.5-flash-lite",
		contents=prompt,
		config=types.GenerateContentConfig(
			temperature=0.0,
			max_output_tokens=400,
			thinking_config=types.ThinkingConfig(thinking_budget=0),
		),
	)
	return response.text


if question:
	# Session limit
	if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
		st.warning(f"⚠️ Demo limiti doldu ({MAX_QUERIES_PER_SESSION} soru). Sayfayı yenileyin.")
		st.stop()

	# Daily global limit
	daily = check_daily_limit()
	if daily["count"] >= MAX_DAILY_QUERIES:
		st.error("⚠️ Şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.")
		st.stop()

	with st.spinner("Cevap aranıyor..."):
		results = collection.query(query_texts=[question], n_results=8)
		context = "\n".join(f"- {d}" for d in results["documents"][0])
		answer = ask_llm(question, context)
		st.session_state.query_count += 1
		daily["count"] += 1

	st.markdown("### Cevap")
	st.write(answer)

	with st.expander("Kullanılan context"):
		docs = results["documents"][0]
		metas = results["metadatas"][0]
		for i, (doc, meta) in enumerate(zip(docs, metas)):
			page = meta.get("page", "?")
			st.markdown(f"**[{i+1}] Sayfa {page}**")
			st.text(doc[:300] + "...")
