from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq


#Step 1a - Indexing

video_id = "sRH-XlBE2cQ"

try: 
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No caption available for this video")


#Step 2 Chunking 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript])

print(len(chunks))