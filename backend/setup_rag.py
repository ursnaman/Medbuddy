

from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "medbuddy-knowledge"

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index: {INDEX_NAME}")
else:
    print(f"Index already exists: {INDEX_NAME}")

medical_docs = [
    """Paracetamol (Acetaminophen):
    Uses: Fever, mild to moderate pain, headache, body aches.
    Adult dose: 500mg to 1000mg every 4-6 hours. Maximum 4000mg per day.
    Children dose: 10-15mg per kg body weight every 4-6 hours.
    Side effects: Safe at normal doses. Overdose causes liver damage.
    Available OTC: Yes, no prescription needed.
    Brand names in India: Crocin, Dolo 650, Calpol, Tylenol.""",

    """Ibuprofen:
    Uses: Pain, fever, inflammation, headache, menstrual cramps.
    Adult dose: 200-400mg every 4-6 hours. Maximum 1200mg per day OTC.
    Side effects: Stomach upset, avoid on empty stomach.
    Warning: Avoid in dengue fever - can cause bleeding.
    Available OTC: Yes.
    Brand names in India: Brufen, Ibugesic, Advil.""",

    """Common Cold:
    Symptoms: Runny nose, sneezing, sore throat, mild fever, cough.
    Cause: Rhinovirus - viral infection.
    Duration: 7-10 days naturally.
    Treatment: Rest, plenty of fluids, steam inhalation.
    Medicines: Paracetamol for fever, antihistamines for runny nose.
    Important: Antibiotics do NOT work for viral infections.
    See doctor if: Fever above 103F, symptoms worsen after 10 days.""",

    """Dengue Fever:
    Symptoms: High fever 104F, severe headache, pain behind eyes,
    joint and muscle pain, skin rash, mild bleeding from nose or gums.
    Cause: Aedes mosquito bite.
    Warning signs (go to emergency): Severe abdominal pain,
    persistent vomiting, bleeding gums, blood in urine or stool,
    rapid breathing, fatigue.
    Treatment: No specific antiviral medicine exists.
    Use Paracetamol for fever - DO NOT use Ibuprofen or Aspirin.
    Stay hydrated, monitor platelet count.
    Hospitalization required if warning signs appear.""",

    """Typhoid Fever:
    Symptoms: Prolonged high fever 103-104F, weakness, stomach pain,
    headache, loss of appetite, sometimes rash.
    Cause: Salmonella typhi bacteria from contaminated food or water.
    Duration: 3-4 weeks if untreated.
    Treatment: Antibiotics required - ciprofloxacin or azithromycin.
    Must see doctor - prescription medicines needed.
    Prevention: Typhoid vaccine, safe drinking water, proper hygiene.""",

    """Hypertension (High Blood Pressure):
    Normal: Below 120/80 mmHg.
    High: Above 140/90 mmHg.
    Symptoms: Often no symptoms - called silent killer.
    Risk factors: Obesity, stress, salt intake, smoking, alcohol.
    Lifestyle changes: Reduce salt, exercise, lose weight, quit smoking.
    Medicines: Amlodipine, Atenolol, Losartan - prescription only.
    Emergency: BP above 180/120 - go to hospital immediately.""",

    """Diabetes Type 2:
    Symptoms: Frequent urination, excessive thirst, unexplained weight loss,
    blurred vision, slow healing wounds, frequent infections.
    Diagnosis: Fasting blood sugar above 126 mg/dL.
    Treatment: Lifestyle changes first, then medicines like Metformin.
    Monitoring: Check blood sugar regularly.
    Complications: Heart disease, kidney damage, vision loss, nerve damage.
    Must see doctor: Requires proper diagnosis and prescription.""",

    """Dehydration:
    Symptoms: Thirst, dark yellow urine, dry mouth, dizziness,
    fatigue, less frequent urination.
    Causes: Vomiting, diarrhea, excessive sweating, fever.
    Treatment: ORS (Oral Rehydration Solution) - mix 1 liter water
    with 6 teaspoons sugar and half teaspoon salt.
    ORS packets available OTC: Electral, Enerlyte.
    See doctor if: No urination for 8 hours, confusion, sunken eyes.""",

    """Gastroenteritis (Food Poisoning / Stomach Flu):
    Symptoms: Nausea, vomiting, diarrhea, stomach cramps, fever.
    Causes: Contaminated food or water, viral or bacterial infection.
    Treatment: Rest, oral rehydration, bland diet (rice, banana, toast).
    Medicines: ORS for hydration, Ondansetron for vomiting (OTC).
    See doctor if: Blood in stool, fever above 102F, symptoms over 3 days,
    signs of severe dehydration.""",

    """Migraine:
    Symptoms: Severe throbbing headache usually on one side, nausea,
    vomiting, sensitivity to light and sound. Can last 4-72 hours.
    Triggers: Stress, lack of sleep, certain foods, hormonal changes.
    Treatment: Rest in dark quiet room, cold compress on forehead.
    Medicines: Ibuprofen or Paracetamol at first sign of migraine.
    Prescription options: Sumatriptan - see doctor for recurrent migraines.
    See doctor if: Worst headache of your life, fever with headache,
    headache after head injury.""",
]


print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.create_documents(medical_docs)
print(f"Created {len(chunks)} chunks from {len(medical_docs)} documents")


print("Loading embedding model (first time takes 2 minutes)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded!")


print("Uploading medical knowledge to Pinecone...")
index = pc.Index(INDEX_NAME)

texts = [doc.page_content for doc in chunks]
vectors = embeddings.embed_documents(texts)

batch_size = 50
for i in range(0, len(vectors), batch_size):
    batch_vectors = vectors[i:i+batch_size]
    batch_texts = texts[i:i+batch_size]
    batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_vectors))]

    index.upsert(
        vectors=[
            {
                "id": batch_ids[k],
                "values": batch_vectors[k],
                "metadata": {"text": batch_texts[k]}
            }
            for k in range(len(batch_vectors))
        ]
    )

print(f"Uploaded {len(vectors)} chunks to Pinecone!")
print("Medical knowledge database is ready!")
print(f"Index stats: {index.describe_index_stats()}")
