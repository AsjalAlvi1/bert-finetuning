from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from io import StringIO
import pandas as pd

# Initialize FastAPI appp
app = FastAPI()

# Database initialization
DATABASE_URL = "sqlite:///./sentiment_analysis.db"
database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define SQLAlchemy model for sentiment analysis data
class SentimentData(Base):
    __tablename__ = "sentiment_data"
    id = Column(Integer, primary_key=True, index=True)
    comment_id = Column(Integer)
    campaign_id = Column(Integer)
    text = Column(Text)
    sentiment = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

class BertClassifier(nn.Module):
     def __init__(self, freeze_bert=False):
      super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
      D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
      self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
      self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
      if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

     def forward(self, input_ids, attention_mask):
      # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


# Load
device = torch.device('cpu')
model_path = "./Downloads/bert.pth"

# Load sentiment analysis model
custom_bert_model = BertClassifier()
custom_bert_model.load_state_dict(torch.load(model_path, map_location=device))
custom_bert_model.eval()  # Set model to evaluation mode
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = custom_bert_model  # Use the custom BERT model
model.eval()

# Function to perform sentiment analysis inference
def perform_sentiment_analysis(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    #print(inputs)
    with torch.no_grad():
        outputs = model(inputs.input_ids,inputs.attention_mask)
        logits = outputs
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label

# Pydantic models for request and response data
class SentimentPrediction(BaseModel):
    text: str

class SentimentDataCreate(BaseModel):
    comment_id: int
    campaign_id: int
    text: str
    sentiment: str

class SentimentDataUpdate(BaseModel):
    comment_id: int
    campaign_id: int
    text: str
    sentiment: str

# Endpoints for CRUD operations
@app.post("/predict")
async def predict_sentiment(data: SentimentPrediction) -> dict:
    predicted_label = perform_sentiment_analysis(data.text)
    return {"sentiment": "positive" if predicted_label == 1 else "negative"}

@app.post("/insert")
async def insert_sentiment_data(data: SentimentDataCreate) -> dict:
    query = SentimentData.__table__.insert().values(**data.dict())
    last_record_id = await database.execute(query)
    return {**data.dict(), "id": last_record_id}

@app.delete("/delete/{comment_id}")
async def delete_sentiment_data(comment_id: int):
    query = SentimentData.__table__.delete().where(SentimentData.comment_id == comment_id)
    await database.execute(query)
    return {"message": "Sentiment data deleted successfully"}

@app.put("/update/{comment_id}")
async def update_sentiment_data(comment_id: int, data: SentimentDataUpdate) -> dict:
    query = (
        SentimentData.__table__.update()
        .where(SentimentData.comment_id == comment_id)
        .values(**data.dict())
    )
    await database.execute(query)
    return {**data.dict(), "comment_id": comment_id}

@app.post("/bulk_insert")
async def bulk_insert(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    print(df)
    # Compute sentiment using the model
    df['sentiment'] = df['comment_description'].apply(perform_sentiment_analysis)
    df=df.rename(columns={"comment_description":"text"})
    df['sentiment'] = df['sentiment'].map({1: 'positive', 0: 'negative'})

    # Bulk insert into the SQLite database
    query = SentimentData.__table__.insert()
    values = df.to_dict(orient='records')
    await database.execute_many(query=query, values=values)
