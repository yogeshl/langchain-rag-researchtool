# langchain-rag-researchtool

conda create -p  venv  python=3.10 
conda activate venv/
pip install -r requirements.txt

streamlit run main.py


## Architecture
![Architecture](architecture.png)

## Load content, Embedding and store in DB
![alt text](image.png)

## Do similarity search, get the chunks and create prompt and send to Open AI LLM
![alt text](image-1.png)

## Splits, Merged Splits and Overlap chunks
![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)