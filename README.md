
[Also Read this full blog on Global News One](https://bit.ly/4191gHh)

### Step 1: **Import Libraries**

```python
import re
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```

#### Explanation of Libraries:
1. **re (Regular Expressions)**:
   - **Purpose**: Provides functions for text processing, allowing pattern matching and text replacement.
   - **Why Used**: Used for preprocessing questions by removing special characters and converting text to lowercase, making the text uniform.

2. **pandas (Python Data Analysis Library)**:
   - **Abbreviation**: `pd`
   - **Purpose**: A powerful data manipulation library for data cleaning, transformation, and analysis.
   - **Why Used**: Used to convert the list of classified questions into a DataFrame for easy review and export to a CSV file.

3. **torch (PyTorch)**:
   - **Abbreviation**: `torch`
   - **Purpose**: A machine learning library that provides tensor operations and supports GPU acceleration.
   - **Why Used**: Used to load and handle the BERT model and perform tensor computations on GPU for faster processing.

4. **transformers (Transformers by Hugging Face)**:
   - **Modules Used**: `BertTokenizer` and `BertModel`
   - **Purpose**: Provides pre-trained transformer models and tokenizers, including BERT.
   - **Why Used**: Used to load the pre-trained BERT model and tokenizer to convert questions into embeddings based on their semantic meaning.

5. **sklearn (Scikit-Learn)**:
   - **Modules Used**: `KMeans` for clustering and `PCA` for dimensionality reduction
   - **Purpose**: A comprehensive machine learning library with algorithms for clustering, classification, regression, and more.
   - **Why Used**: Used for clustering embeddings into groups (K-means) and reducing dimensions for visualization (PCA).

6. **matplotlib (Matplotlib for Plotting)**:
   - **Module Used**: `pyplot` (abbreviated as `plt`)
   - **Purpose**: A plotting library that allows for the creation of static, animated, and interactive visualizations.
   - **Why Used**: Used to visualize clustered questions after dimensionality reduction, aiding in visual inspection of clustering quality.

---

### Step 2: **Load the Dataset**

```python
import json

# Load the dataset (assuming the file path is '/content/questions.json')
with open('/content/questions.json', 'r') as file:
    data = json.load(file)

# Extract question texts
questions = [item['question_text'] for item in data]
print(f"Total questions loaded: {len(questions)}")
print(f"Sample question: {questions[0]}")
```

#### Explanation:
- **json (JavaScript Object Notation)**: A lightweight format for storing and transporting data, commonly used for data interchange.
- **Why Used**: Reads the JSON file containing questions, extracting the `question_text` field from each entry to store in a list called `questions`.

---

### Step 3: **Preprocess the Questions**

```python
# Function to clean question text
def preprocess_text(text):
    # Remove special characters and lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

# Apply preprocessing to each question
questions_cleaned = [preprocess_text(q) for q in questions]
print(f"Sample cleaned question: {questions_cleaned[0]}")
```

#### Explanation:
- **Purpose of Preprocessing**: 
   - Text preprocessing cleans each question, making it easier to create consistent embeddings. By removing special characters and converting to lowercase, we ensure that only the core words contribute to the model, reducing noise.
- **Function Workflow**:
   - **Special Characters Removal**: Uses `re.sub()` to remove all characters that aren’t letters or whitespace.
   - **Lowercasing**: Converts all text to lowercase to ensure uniformity.

---

### Step 4: **Load BERT Model and Tokenizer**

```python
# Load pre-trained BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check if CUDA is available and move model to GPU if so
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

#### Explanation:
- **BERT (Bidirectional Encoder Representations from Transformers)**:
   - A transformer model pre-trained on a large corpus to understand language semantics, allowing it to generate meaningful embeddings.
- **transformers (Transformers Library)**:
   - **BertTokenizer**: Converts text into tokens that BERT understands.
   - **BertModel**: Loads the pre-trained model, enabling us to extract embeddings for each question.
- **CUDA (Compute Unified Device Architecture)**:
   - A parallel computing platform allowing PyTorch to leverage GPU for faster computation.
- **Why Used**: Tokenizes questions and generates embeddings using BERT’s trained parameters. Using GPU speeds up computation.

---

### Step 5: **Generate BERT Embeddings**

```python
def get_bert_embedding(text):
    # Tokenize and encode text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU if available

    # Get BERT model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the mean of the last hidden state as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Generate embeddings for all questions
question_embeddings = [get_bert_embedding(q) for q in questions_cleaned]
print("Generated embeddings for all questions.")
```

#### Explanation:
- **Tokenization and Encoding**: Converts text to tokens and then encodes them as PyTorch tensors for BERT.
- **Embedding Extraction**:
   - `model(**inputs)`: Passes tokens through BERT.
   - `outputs.last_hidden_state`: Accesses the hidden state of the last BERT layer for each token in the question.
   - **Mean Pooling**: Averages the last hidden states to create a single embedding vector, representing the question’s semantic meaning.
- **Why Used**: This process converts questions into dense embeddings, capturing each question’s meaning in a way that K-means can cluster.

---

### Step 6: **Cluster Embeddings Using K-means**

```python
# Define the number of clusters based on suggested categories (you can adjust this number)
n_clusters = 7

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(question_embeddings)

# Check a few cluster assignments
print(f"Cluster assignment for first 5 questions: {clusters[:5]}")
```

#### Explanation:
- **K-means Clustering**:
   - A widely used clustering algorithm that partitions data into `k` clusters based on similarity.
   - **n_clusters**: The number of clusters (categories) we want to find.
   - **fit_predict**: Assigns each question to the nearest cluster based on embedding similarity.
- **Why Used**: Organizes questions into categories by grouping similar embeddings together, simplifying the classification process.

---

### Step 7: **Map Clusters to Descriptive Categories**

```python
# Define a mapping of cluster labels to descriptive categories
cluster_mapping = {
    0: "Material Specification",
    1: "Document Request",
    2: "Installation Requirements",
    3: "Responsibility and Ownership",
    4: "Compliance and Code Requirements",
    5: "Project Status and Tracking",
    6: "Equipment and Component Inquiry"
}

# Assign each question a category based on its cluster label
classified_questions = [
    {"question": q, "classification": cluster_mapping[clusters[i]]}
    for i, q in enumerate(questions)
]
```

#### Explanation:
- **Manual Mapping**: After clustering, each cluster is given a descriptive name based on observed question types.
- **Purpose**: Improves interpretability by assigning meaningful names to clusters, making the classification results more understandable.

---

### Step 8: **Display and Verify Clustered Questions**

```python
# Convert to DataFrame for easier review
classified_df = pd.DataFrame(classified_questions)
print(classified_df.head())

# Display some questions from each category for verification
for cluster_id, category in cluster_mapping.items():
    print(f"Category: {category}")
    sample_questions = classified_df[classified_df['classification'] == category].head(5)
    for idx, row in sample_questions.iterrows():
        print(f" - {row['question']}")
    print("\n")
```

#### Explanation:
- **Data Conversion**: Converts results to a DataFrame for organized presentation and easy review.
- **Sample Verification**: Displays a few sample questions from each category to manually verify the accuracy of clustering.

---

### Step 9: **Dimensionality Reduction and Visualization**

```python
from sklearn.decomposition import PCA

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(question_embeddings)

# Plot clusters
plt.figure(figsize=(10, 8))
for cluster_id in range(n_clusters):
   

 cluster_points = reduced_embeddings[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_mapping[cluster_id])

plt.legend()
plt.title("Question Clusters by Sense")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
```

#### Explanation:
- **PCA (Principal Component Analysis)**: Reduces high-dimensional embeddings to 2D, making visualization possible.
- **Plotting**: Visualizes clusters by plotting each question’s reduced embedding, color-coded by category.
- **Why Used**: Helps inspect how well the questions separate into clusters, providing a visual validation.

---

### Step 10: **Save Classified Results to CSV**

```python
# Save the classified questions to a CSV file
classified_df.to_csv('classified_questions.csv', index=False)
print("Classified questions saved to 'classified_questions.csv'")
```

#### Explanation:
- **Exporting Data**: Saves the final categorized results to a CSV file for easy access and further analysis.

---

### Steps 11-14: **Evaluate Clustering Quality with Internal Metrics**

Each of these metrics provides insight into clustering quality:
1. **Inertia (Within-cluster sum of squares)**: Lower values mean tighter clusters.
2. **Silhouette Score**: Measures how similar each question is to its cluster versus other clusters (higher is better).
3. **Calinski-Harabasz Index**: Measures between-cluster dispersion (higher is better).
4. **Davies-Bouldin Index**: Measures similarity between clusters (lower is better).

---

### Step 15: **t-SNE Visualization (Optional)**

This provides another dimensionality reduction method that better captures non-linear relationships, offering improved cluster separation visualization over PCA.

**t-SNE** stands for **t-distributed Stochastic Neighbor Embedding**.

### Explanation of t-SNE and Its Purpose
- **t-distributed**: Refers to the statistical distribution applied in the algorithm, which helps it handle large datasets by making distant points less likely to attract each other. 
- **Stochastic**: Implies that the method is based on randomness and probabilistic calculations. This randomness helps to focus on preserving local relationships (clusters) rather than global structure.
- **Neighbor Embedding**: Refers to the algorithm's goal to preserve local structure by mapping nearby points in high-dimensional space to nearby points in lower-dimensional space.

### Purpose of t-SNE in This Project
- **Dimensionality Reduction**: t-SNE is used to reduce high-dimensional data (such as BERT embeddings) to 2D or 3D, enabling visualization.
- **Capturing Non-linear Relationships**: Unlike linear methods like PCA, t-SNE captures non-linear relationships in the data, often providing clearer separations between clusters in a visual plot.
- **Local Structure Preservation**: t-SNE focuses on maintaining the relative distances between neighboring points, making it effective for visualizing clusters where local groupings are meaningful.

In this project, **t-SNE helps visualize clusters of question embeddings** by showing how different groups (based on semantic similarity) are separated in the reduced 2D space.
