
conda activate genai
.\.venv\Scripts\Activate.ps1  

# Terminal 1: Start the Backend (FastAPI)
 cd C:\Users\ps302\OneDrive\Desktop\Hydrology
 python -m uvicorn api.main:app --reload

# Terminal 2: Start the Frontend (Streamlit)
 cd C:\Users\ps302\OneDrive\Desktop\Hydrology
 streamlit run app.py

# Terminal 3: PostresSaver- start
 docker compose up -d

# Data file link:
https://drive.google.com/drive/folders/1dtrUnhkgqphh1hzxRxO1FFkHeMsp4X2L?usp=sharing 


# to delete complete memory of langdb
docker exec -it langgraph_postgres psql -U langgraph -d langgraph_db -c "TRUNCATE checkpoints, checkpoint_writes, checkpoint_blobs RESTART IDENTITY CASCADE;"

# to run the app
streamlit run src/Flood_Model/flood_app/app.py

# test_pipeline
python src/Flood_Model/flood_app/test_pipeline.py



 Wait until you see Uvicorn running on http://127.0.0.1:8000.

# test retriver
python -m vectorStore.test_retriever

 # Qudrant-dashboard
 http://localhost:6333/dashboard


 # Quadrant - stop
 docker stop hydrology-qdrant

# 1. Enter DB
docker exec -it langgraph_postgres psql -U langgraph -d langgraph_db

# 2. List tables
\dt

# 3. See threads
SELECT thread_id FROM checkpoints;

# 4. See memory
SELECT * FROM checkpoint_writes;

# 5. Exit
\q


# Starting Everything Next Time
# Next time you restart your computer.
# Just run:

docker compose up -d

# This starts PostgreSQL again.
# Because we used Docker Volume, the data stays.
# Verify DB is Running Run:

docker ps

# You should see:

langgraph_postgres




